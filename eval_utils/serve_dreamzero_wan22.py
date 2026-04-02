# [CC] 本文件实现了DreamZero 5B模型的WebSocket推理服务端，用于将训练好的DreamZero策略部署为可远程调用的服务。
# [CC] 客户端通过WebSocket发送观测数据（图像、关节状态、语言指令），服务端返回动作预测结果。
"""
Serve the DreamZero 5B implementation (Wan2.2-TI2V-5B) over the websocket policy server.

This is the 5B model: Wan2.2 diffusion backbone, 48-channel VAE38, frame_seqlen=50 (160×320
latent 10×20). Inference is causal with KV caching: first request in a session uses 1 frame
and warms the cache; subsequent requests use FRAMES_PER_CHUNK=4 frames and append to the cache.
On session_id change (or explicit reset), buffers and action_head.current_start_frame are cleared.

The checkpoint at model_path should be DreamZero with Wan22 5B (model/dreamzero/action_head=
wan_flow_matching_action_tf_wan22, data droid_relative_wan22 → 160×320). GrootSimPolicy loads
that checkpoint and runs inference; it is the correct policy class for DreamZero.

Usage (single GPU):

  torchrun --nproc_per_node=1 eval_utils/serve_dreamzero_wan22.py --model_path ./checkpoints/dreamzero_droid_wan22_smoke --port 8000

  # Or single process:
  python eval_utils/serve_dreamzero_wan22.py --model_path ./checkpoints/dreamzero_droid_wan22_smoke --port 8000

Client: send observations per PolicyServerConfig (policy_server.py). Video is resized to the
checkpoint's expected resolution (e.g. 180×320) so the eval transform accepts it; the 5B action
head resizes to 160×320 internally. Override with --image_height/--image_width if needed.
Response is an action chunk (N, 8). Use session_id for episode boundaries.
"""

# [CC] 导入标准库和第三方依赖
import datetime
import logging
import os
import sys

import imageio

logger = logging.getLogger(__name__)

import cv2
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import tyro

# [CC] 防止torch.compile在推理服务中因输入形状变化导致重编译次数超限而报错。
# [CC] flow scheduler的multistep_uni_p_bh_update在不同batch size、step_index、order下会触发重编译，
# [CC] 因此需要提高缓存和重编译上限。
_dynamo = torch._dynamo.config
if hasattr(_dynamo, "cache_size_limit"):
    _dynamo.cache_size_limit = 1000
if hasattr(_dynamo, "recompile_limit"):
    _dynamo.recompile_limit = 800
if hasattr(_dynamo, "accumulated_cache_size_limit"):
    _dynamo.accumulated_cache_size_limit = 1000
if hasattr(_dynamo, "accumulated_recompile_limit"):
    _dynamo.accumulated_recompile_limit = 2000
from pathlib import Path
from tianshou.data import Batch

# [CC] 将仓库根目录添加到Python路径，以便导入项目内部模块
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi_client.base_policy import BasePolicy

from eval_utils.policy_server import WebsocketPolicyServer, PolicyServerConfig
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # [CC] DreamZero的策略推理类，负责加载checkpoint并执行前向推理
from groot.vla.data.schema import EmbodimentTag  # [CC] 机器人硬件平台标签（如oxe_droid）
from groot.vla.data.transform import ComposedModalityTransform  # [CC] 多模态数据预处理变换的组合类


# [CC] DreamZero Wan 5B 训练时使用的默认分辨率（droid_relative_wan22数据集）
DEFAULT_IMAGE_HEIGHT = 160
DEFAULT_IMAGE_WIDTH = 320
# [CC] 因果分块推理每次处理的帧数，与5B模型的num_frame_per_block一致
FRAMES_PER_CHUNK = 4


def _get_expected_video_resolution(policy: GrootSimPolicy) -> tuple[int, int]:
    # [CC] 从策略的eval_transform中读取checkpoint元数据里记录的视频分辨率。
    # [CC] 元数据中分辨率格式为(width, height)，本函数返回(height, width)以符合resize惯例。
    """Get (height, width) the policy's eval_transform expects for video (from checkpoint
    metadata). Resolution in metadata is (width, height); we return (height, width) for resize.
    DreamZero Wan 5B (droid_relative_wan22) uses 160×320; other configs may use e.g. 180×320.
    """
    eval_transform = getattr(policy, "eval_transform", None)
    if eval_transform is None:
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    if not isinstance(eval_transform, ComposedModalityTransform):
        return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)
    # [CC] 遍历所有变换，查找包含original_resolutions属性的变换（记录了训练时的分辨率信息）
    for t in eval_transform.transforms:
        if hasattr(t, "original_resolutions") and getattr(t, "original_resolutions", None):
            res = t.original_resolutions
            if res:
                # [CC] original_resolutions的值格式是(width, height)，需要交换为(height, width)
                w, h = next(iter(res.values()))
                return (int(h), int(w))
    return (DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH)


def _resize_frames_to_resolution(frames: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # [CC] 将视频帧缩放到目标分辨率。支持单帧(H,W,C)和多帧(T,H,W,C)两种输入格式。
    """Resize video frames to (target_h, target_w). Accepts (H,W,C) or (T,H,W,C)."""
    if frames.ndim == 3:
        # [CC] 单帧情况：仅在尺寸不匹配时执行缩放
        if (frames.shape[0], frames.shape[1]) != (target_h, target_w):
            frames = cv2.resize(frames, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return frames
    # [CC] 多帧情况：逐帧缩放后重新堆叠
    out = np.stack(
        [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR) for f in frames],
        axis=0,
    )
    return out


def _maybe_init_distributed():
    # [CC] 初始化PyTorch分布式进程组（GrootSimPolicy要求分布式环境已初始化）。
    # [CC] 单GPU推理时也需要初始化，world_size设为1即可。
    """Initialize process group for single-GPU or multi-GPU. Required by GrootSimPolicy."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


# [CC] 模态键名映射：将客户端发送的观测键名转换为模型内部使用的键名，按机器人平台分类。
# [CC] 客户端发送的键名格式为 observation/xxx，模型内部使用 video.xxx / state.xxx / annotation.xxx 格式。
VIDEO_KEY_MAPPING = {
    "oxe_droid": {
        # [CC] 外部相机0 -> 模型的exterior_image_1_left
        "observation/exterior_image_0_left": "video.exterior_image_1_left",
        # [CC] 外部相机1 -> 模型的exterior_image_2_left
        "observation/exterior_image_1_left": "video.exterior_image_2_left",
        # [CC] 腕部相机 -> 模型的wrist_image_left
        "observation/wrist_image_left": "video.wrist_image_left",
    },
}
# [CC] 状态键名映射：关节位置和夹爪位置
STATE_KEY_MAPPING = {
    "oxe_droid": ("state.joint_position", "state.gripper_position"),
}
# [CC] 语言指令键名映射：自然语言任务描述
LANGUAGE_KEY_MAPPING = {
    "oxe_droid": "annotation.language.action_text",
}


# [CC] DreamZero Wan2.2 5B策略封装类。
# [CC] 负责将客户端的观测数据格式转换为模型所需的Batch格式，调用GrootSimPolicy进行推理，
# [CC] 并将模型输出的动作转换为标准的(N, 8)数组返回给客户端。
# [CC] 支持因果分块推理：首次调用使用1帧预热KV缓存，后续调用每次使用4帧。
# [CC] 通过session_id管理会话边界，会话切换时自动重置所有缓冲区。
class DreamZeroWan225BPolicy(BasePolicy):
    """
    Wraps GrootSimPolicy for the DreamZero 5B implementation (Wan2.2-TI2V-5B).

    Converts roboarena observation/action format to DROID/Batch. Video is resized to the
    resolution expected by the policy's eval_transform (from checkpoint metadata) so
    VideoToTensor validation passes. The 5B action head then resizes to 160×320 internally.
    First call in a session uses 1 frame; later calls use 4 frames (FRAMES_PER_CHUNK).
    Session reset clears frame buffers and action_head.current_start_frame.
    """

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        image_height: int,
        image_width: int,
        embodiment_tag: str = "oxe_droid",
        save_video_pred: bool = False,
        video_output_dir: str = "./video_pred_output",
    ):
        # [CC] 初始化DreamZero策略封装器。
        # [CC] 参数说明：
        # [CC]   groot_policy: 已加载checkpoint的GrootSimPolicy实例
        # [CC]   image_height/image_width: 输入视频帧的目标分辨率
        # [CC]   embodiment_tag: 机器人平台标识，决定键名映射
        # [CC]   save_video_pred: 是否保存模型预测的视频（用于可视化调试）
        super().__init__()
        self._policy = groot_policy
        self._image_height = image_height
        self._image_width = image_width
        # [CC] 验证embodiment_tag是否在支持列表中，不支持则回退到oxe_droid
        self._embodiment_tag = (
            embodiment_tag if embodiment_tag in VIDEO_KEY_MAPPING else "oxe_droid"
        )
        # [CC] 为每个视频模态创建帧缓冲区，用于积累历史帧以供因果推理
        video_keys = list(VIDEO_KEY_MAPPING[self._embodiment_tag].values())
        self._frame_buffers = {k: [] for k in video_keys}
        # [CC] 标记是否为会话中的首次调用（首次用1帧预热KV缓存）
        self._is_first_call = True
        self._current_session_id = None
        self._save_video_pred = save_video_pred
        self._video_output_dir = video_output_dir
        # [CC] 累积视频预测的潜变量，用于最终解码保存完整预测视频
        self._video_pred_latents: list[torch.Tensor] = []
        self._current_prompt: str = ""

    def _convert_observation(self, obs: dict) -> dict:
        # [CC] 将客户端发来的观测字典转换为模型所需的Batch输入格式。
        # [CC] 主要完成三件事：1) 视频帧缩放并填入帧缓冲区 2) 提取关节/夹爪状态 3) 提取语言指令。
        """Convert roboarena observation format to model Batch format.
        Incoming frames are resized to the policy's expected (height, width) so
        eval_transform's VideoToTensor check passes.
        """
        # [CC] 第一步：处理视频帧，缩放到目标分辨率并追加到帧缓冲区
        image_key_mapping = VIDEO_KEY_MAPPING[self._embodiment_tag]
        for roboarena_key, model_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    data = _resize_frames_to_resolution(
                        data, self._image_height, self._image_width
                    )
                    # [CC] 4维数组表示多帧(T,H,W,C)，逐帧添加；3维表示单帧(H,W,C)，直接添加
                    if data.ndim == 4:
                        self._frame_buffers[model_key].extend(list(data))
                    else:
                        self._frame_buffers[model_key].append(data)

        # [CC] 第二步：根据是否首次调用决定使用的帧数（首次1帧预热，后续4帧）
        num_frames = 1 if self._is_first_call else FRAMES_PER_CHUNK
        converted = {}
        for model_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # [CC] 缓冲区帧数足够，取最后num_frames帧
                    frames_to_use = buffer[-num_frames:]
                else:
                    # [CC] 缓冲区帧数不足，用第一帧填充到所需数量
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        frames_to_use.insert(0, buffer[0])
                video = np.stack(frames_to_use, axis=0)
                converted[model_key] = video

        # [CC] 第三步：处理关节位置状态，缺失时填零
        state_joint_key, state_gripper_key = STATE_KEY_MAPPING[self._embodiment_tag]
        if "observation/joint_position" in obs:
            joint_pos = np.asarray(obs["observation/joint_position"])
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted[state_joint_key] = joint_pos.astype(np.float64)
        else:
            converted[state_joint_key] = np.zeros((1, 7), dtype=np.float64)

        # [CC] 处理夹爪位置状态，缺失时填零
        if "observation/gripper_position" in obs:
            gripper_pos = np.asarray(obs["observation/gripper_position"])
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted[state_gripper_key] = gripper_pos.astype(np.float64)
        else:
            converted[state_gripper_key] = np.zeros((1,1), dtype=np.float64)

        # [CC] 第四步：提取语言指令（任务描述文本）
        text_prompt = obs.get("prompt", "")
        logger.info("Text prompt: %s", text_prompt)
        if text_prompt:
            self._current_prompt = text_prompt
        lang_key = LANGUAGE_KEY_MAPPING[self._embodiment_tag]
        converted[lang_key] = text_prompt
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        # [CC] 将模型输出的动作字典转换为(N, 8)的numpy数组，其中前7维为关节位置，最后1维为夹爪位置。
        """Convert model action dict to (N, 8) array (7 joint + 1 gripper)."""
        joint_action = None
        gripper_action = None
        # [CC] 通过键名中的关键词匹配来提取关节动作和夹爪动作
        for key, value in action_dict.items():
            if ("joint_position" in key or "joint_pos" in key) and "gripper" not in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        # [CC] 如果没有找到关节动作，返回全零动作
        if joint_action is None:
            return np.zeros((1, 8), dtype=np.float32)
        # [CC] 将torch.Tensor转换为numpy数组
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        N = joint_action.shape[0]
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            # [CC] 夹爪动作只取第一维（可能包含多余维度）
            if gripper_action.shape[-1] > 1:
                gripper_action = gripper_action[..., :1]
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        # [CC] 拼接关节动作和夹爪动作，形成完整的(N, 8)动作数组
        return np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

    def infer(self, obs: dict) -> np.ndarray:
        # [CC] 核心推理方法：接收客户端观测，返回动作预测。
        # [CC] 流程：检查会话边界 -> 转换观测格式 -> 因果推理 -> 转换动作格式 -> 返回结果。

        # [CC] 检测session_id变化，如果会话切换则重置所有缓冲区
        session_id = obs.get("session_id")
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                self.reset({})
            self._current_session_id = session_id

        # [CC] 将客户端观测转换为模型输入格式
        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        # [CC] 调用GrootSimPolicy的因果推理接口，返回动作结果和视频预测潜变量
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        # [CC] 如果开启了视频预测保存，累积潜变量供后续解码
        if self._save_video_pred and video_pred is not None:
            self._video_pred_latents.append(video_pred.detach())
        # [CC] 从结果Batch中提取所有以"action."开头的属性，组成动作字典
        action_dict = {}
        action_chunk_dict = result_batch.act
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        action = self._convert_action(action_dict)
        # [CC] 首次调用完成后标记为非首次，后续将使用FRAMES_PER_CHUNK帧进行推理
        if self._is_first_call:
            self._is_first_call = False
        return action

    def _save_predicted_video(self) -> None:
        # [CC] 将累积的视频预测潜变量通过VAE解码器解码为像素帧，并保存为mp4视频文件。
        # [CC] 在会话重置时被调用，用于可视化模型预测的未来视频。
        """Decode accumulated video prediction latents through the VAE and save as mp4."""
        if not self._video_pred_latents:
            return
        try:
            from einops import rearrange

            action_head = self._policy.trained_model.action_head
            # [CC] 将所有chunk的潜变量在时间维度(dim=2)上拼接
            latents = torch.cat(self._video_pred_latents, dim=2)
            # [CC] 使用VAE解码器将潜变量解码为像素帧，支持分块解码以节省显存
            with torch.no_grad():
                frames = action_head.vae.decode(
                    latents,
                    tiled=action_head.tiled,
                    tile_size=(action_head.tile_size_height, action_head.tile_size_width),
                    tile_stride=(action_head.tile_stride_height, action_head.tile_stride_width),
                )
            # [CC] 将(B,C,T,H,W)格式转换为(B,T,H,W,C)，取第一个batch
            frames = rearrange(frames, "B C T H W -> B T H W C")[0]
            # [CC] 将[-1,1]范围的浮点像素值转换为[0,255]的uint8格式
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)

            os.makedirs(self._video_output_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
            n_latent_frames = latents.shape[2]
            existing = [f for f in os.listdir(self._video_output_dir) if f.endswith(".mp4")]
            # [CC] 生成安全的文件名：去除特殊字符，截断过长的prompt
            safe_prompt = self._current_prompt.replace(" ", "_")
            safe_prompt = "".join(c for c in safe_prompt if c.isalnum() or c in "_-.")
            if len(safe_prompt) > 80:
                safe_prompt = safe_prompt[:80]
            if not safe_prompt:
                safe_prompt = "no_prompt"
            # [CC] 文件名格式：序号_任务描述_时间戳.mp4
            output_path = os.path.join(
                self._video_output_dir,
                f"{len(existing):06}_{safe_prompt}_{timestamp}.mp4",
            )
            imageio.mimsave(output_path, list(frames), fps=5, codec="libx264")
            logger.info("Saved video prediction (%d frames) to %s", len(frames), output_path)
        except Exception as e:
            logger.warning("Failed to save video prediction: %s", e)

    def reset(self, reset_info: dict) -> None:
        # [CC] 重置策略状态，在会话结束或切换时调用。
        # [CC] 清空帧缓冲区、视频预测潜变量、语言指令，并重置action_head的帧计数器。
        # [CC] 如果开启了视频预测保存，在重置前先保存当前会话的预测视频。
        if self._save_video_pred:
            self._save_predicted_video()
        self._video_pred_latents.clear()
        self._current_prompt = ""
        for key in self._frame_buffers:
            self._frame_buffers[key] = []
        self._is_first_call = True
        self._current_session_id = None
        # [CC] 重置action_head的因果推理帧计数器，使下次推理从第0帧开始
        if hasattr(self._policy.trained_model, "action_head") and hasattr(
            self._policy.trained_model.action_head, "current_start_frame"
        ):
            self._policy.trained_model.action_head.current_start_frame = 0


def main(
    model_path: str = "./checkpoints/dreamzero_droid_wan22_smoke",
    embodiment_tag: str = "oxe_droid",
    tokenizer_path: str | None = None,
    port: int = 8000,
    host: str = "0.0.0.0",
    image_height: int | None = None,
    image_width: int | None = None,
    save_video_pred: bool = False,
    video_output_dir: str = "./video_pred_output",
) -> None:
    # [CC] 主函数：初始化分布式环境、加载模型、创建策略封装器、启动WebSocket推理服务。
    # [CC] 参数可通过命令行传入（由tyro.cli解析）。
    logging.basicConfig(level=logging.INFO, force=True)

    # [CC] 初始化分布式环境和设备网格（单GPU推理时mesh_shape=(1,)）
    _maybe_init_distributed()
    device_mesh = init_device_mesh("cuda", mesh_shape=(1,), mesh_dim_names=("ip",))

    # [CC] 加载DreamZero策略模型（从checkpoint恢复权重和配置）
    logger.info("Loading DreamZero Wan22 policy from %s (embodiment=%s)", model_path, embodiment_tag)
    checkpoint_name = os.path.basename(model_path.rstrip("/"))
    video_output_dir = os.path.join(video_output_dir, checkpoint_name)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        tokenizer_path_override=tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )
    # [CC] 确定视频输入分辨率：优先使用命令行指定的值，否则从checkpoint元数据中读取
    if image_height is not None and image_width is not None:
        h, w = image_height, image_width
        logger.info("Using CLI video resolution: %dx%d", h, w)
    else:
        h, w = _get_expected_video_resolution(policy)
        logger.info("Using checkpoint video resolution: %dx%d (HxW)", h, w)
    # [CC] 创建DreamZero策略封装器，负责格式转换和推理调度
    wrapper = DreamZeroWan225BPolicy(
        groot_policy=policy,
        image_height=h,
        image_width=w,
        embodiment_tag=embodiment_tag,
        save_video_pred=save_video_pred,
        video_output_dir=video_output_dir,
    )

    # [CC] 配置WebSocket策略服务器：指定图像分辨率、相机配置、动作空间等
    server_config = PolicyServerConfig(
        image_resolution=(h, w),
        needs_wrist_camera=True,          # [CC] 需要腕部相机
        n_external_cameras=2,             # [CC] 需要2个外部相机
        needs_stereo_camera=False,        # [CC] 不需要立体相机
        needs_session_id=True,            # [CC] 需要客户端提供session_id以管理会话边界
        action_space="joint_position",    # [CC] 动作空间为关节位置控制
    )
    # [CC] 启动WebSocket服务，进入无限循环等待客户端连接和请求
    logger.info("Starting WebsocketPolicyServer on %s:%d (DreamZero 5B, %dx%d)", host, port, h, w)
    server = WebsocketPolicyServer(
        policy=wrapper,
        server_config=server_config,
        host=host,
        port=port,
    )
    server.serve_forever()


# [CC] 入口点：使用tyro自动解析命令行参数并调用main函数
if __name__ == "__main__":
    tyro.cli(main)
