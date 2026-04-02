# [CC] 本文件实现了一个基于 WebSocket 的分布式策略推理服务器，用于 AR_droid 机器人的实时控制。
# [CC] 主要功能：接收客户端的观测数据，通过分布式推理生成动作指令，并返回给客户端。
# [CC] 支持多 GPU 分布式推理（rank 0 处理 WebSocket 通信，其余 rank 作为 worker 参与计算）。

import dataclasses
import logging
import socket
import asyncio
import os
import http
import logging
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
from tianshou.data import Batch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

# [CC] 导入 roboarena 策略服务器接口，用于标准化的策略服务通信
# Use roboarena policy server interface
from eval_utils.policy_server import WebsocketPolicyServer as RoboarenaServer
from eval_utils.policy_server import PolicyServerConfig

logger = logging.getLogger(__name__)

# [CC] 命令行参数数据类，定义服务器启动时的可配置选项
@dataclasses.dataclass
class Args:
    port: int = 8000  # [CC] WebSocket 服务器监听端口
    timeout_seconds: int = 50000  # [CC] 分布式通信超时时间（秒），默认约14小时
    model_path: str = "./checkpoints/dreamzero"  # [CC] 模型检查点路径
    enable_dit_cache: bool = False  # [CC] 是否启用 DiT 缓存加速推理
    index: int = 0  # [CC] 实验索引，用于区分不同的评估运行
    max_chunk_size: int | None = None  # [CC] 推理时最大 chunk 大小，None 则使用配置默认值


# [CC] AR_droid 策略包装类，将 roboarena 的策略接口适配为 AR_droid 模型所需的格式。
# [CC] 核心职责：
# [CC]   1. 观测格式转换：roboarena 单帧图像 -> AR_droid 多帧视频格式
# [CC]   2. 帧累积管理：跨调用累积图像帧，因为 roboarena 每次发送单帧而 AR_droid 需要多帧视频
# [CC]   3. 动作格式转换：AR_droid 字典格式 -> roboarena 数组格式
# [CC]   4. 分布式推理协调：rank 0 广播数据给其他 worker
class ARDroidRoboarenaPolicy:
    """Wrapper policy that implements roboarena.policy.BasePolicy interface for AR_droid.

    Handles:
    - Observation format conversion (roboarena -> AR_droid format)
    - Frame accumulation across calls (roboarena sends single frames, AR_droid expects multi-frame video)
    - Action format conversion (AR_droid dict -> roboarena array format)
    - Distributed inference coordination
    """

    # [CC] 第一次调用之后，每次推理使用的帧数
    # Number of frames to accumulate after the first call
    FRAMES_PER_CHUNK = 4
    
    # [CC] 初始化策略包装器，设置模型、通信组、输出目录以及各种内部状态缓冲区
    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        signal_group: dist.ProcessGroup,
        output_dir: str | None = None,
    ) -> None:
        self._policy = groot_policy  # [CC] 底层 GrootSimPolicy 模型实例
        self._signal_group = signal_group  # [CC] 用于分布式信号广播的 gloo 进程组
        self._output_dir = output_dir  # [CC] 视频输出保存目录

        # [CC] 每个相机视角的帧缓冲区，用于累积多帧图像
        # Frame buffers for accumulation (per camera view)
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": [],
        }
        self._call_count = 0  # [CC] 推理调用计数器
        self._is_first_call = True  # [CC] 标记是否为首次调用（首次只用1帧）

        # [CC] 会话跟踪：当新会话开始时重置状态
        # Session tracking - reset state when new session starts
        self._current_session_id: str | None = None

        # [CC] 跨时间步累积的视频预测，用于最终保存视频
        # Video across time for saving (similar to original server)
        self.video_across_time = []
        self._msg_index = 0  # [CC] 消息序号计数器

        # [CC] 如果指定了输出目录则创建
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
    
    # [CC] 将 roboarena 格式的观测数据转换为 AR_droid 模型所需的格式。
    # [CC] 主要转换：单帧图像 -> 多帧视频，键名映射（0-indexed -> 1-indexed），状态数据重塑
    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.

        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str

        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}

        # [CC] 图像键名映射：roboarena 使用 0-indexed 命名，AR_droid 使用 1-indexed 命名
        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }
        
        # [CC] 逐相机视角累积帧数据到缓冲区，支持单帧和多帧输入
        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # [CC] 多帧输入 (T, H, W, 3)，逐帧添加到缓冲区
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # [CC] 单帧输入 (H, W, 3)，直接追加
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # [CC] 确定本次推理使用的帧数：首次调用用1帧，后续调用用 FRAMES_PER_CHUNK 帧
        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK

        # [CC] 从累积的帧缓冲区构建视频张量，不足时用首帧填充
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # [CC] 缓冲区帧数充足，取最后 num_frames 帧
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # [CC] 缓冲区帧数不足，用首帧在前面填充至目标帧数
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # [CC] 将帧列表堆叠为 (T, H, W, C) 格式的视频数组
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # [CC] 转换关节位置状态，重塑为 (1, 7) 并转为 float64
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)

        # [CC] 转换夹爪位置状态，重塑为 (1, 1) 并转为 float64
        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)

        # [CC] 转换语言提示文本
        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""

        return converted
    
    # [CC] 将 AR_droid 字典格式的动作转换为 roboarena 的数组格式。
    # [CC] 拼接关节位置 (N,7) 和夹爪位置 (N,1) 为 (N,8) 的动作数组。
    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.

        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)

        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None

        # [CC] 从动作字典中按键名提取关节动作和夹爪动作
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value

        if joint_action is None:
            # [CC] 未找到关节动作时返回全零兜底
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)

        # [CC] 如果是 PyTorch 张量则转为 numpy
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()

        # [CC] 确保关节动作为二维 (N, 7) 格式
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)

        N = joint_action.shape[0]

        # [CC] 处理夹爪动作：转 numpy、重塑维度，缺失时填零
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)

        # [CC] 拼接关节和夹爪动作：(N, 7) + (N, 1) -> (N, 8)
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)

        return action
    
    # [CC] 将观测数据从 rank 0 广播到所有其他 worker 进程。
    # [CC] 使用 pickle 序列化，先广播数据大小，再广播数据本体。
    def _broadcast_batch_to_workers(self, obs: dict) -> None:
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle

        # [CC] 序列化观测字典
        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        # [CC] 先广播数据大小，让 worker 知道要接收多少字节
        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # [CC] 广播序列化后的数据
        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    # [CC] 核心推理方法：接收 roboarena 格式观测，执行分布式前向推理，返回动作数组。
    # [CC] 流程：会话检查 -> 格式转换 -> 广播数据 -> 分布式前向传播 -> 动作转换
    def infer(self, obs: dict) -> np.ndarray:
        """Infer actions from observations.

        Args:
            obs: Observation dict in roboarena format

        Returns:
            action: (N, 8) action array
        """
        # [CC] 检测会话是否切换，如果是新会话则重置所有内部状态
        # Check for session change - reset state if new session
        session_id = obs.get("session_id", None)
        if session_id is not None and session_id != self._current_session_id:
            if self._current_session_id is not None:
                logger.info(f"Session changed from '{self._current_session_id}' to '{session_id}', resetting state")
                # Reset state for new session
                self._reset_state()
            else:
                logger.info(f"New session started: '{session_id}'")
            self._current_session_id = session_id

        self._msg_index += 1
        self._call_count += 1

        # [CC] 将 roboarena 格式观测转换为 AR_droid 格式
        # Convert observation format
        converted_obs = self._convert_observation(obs)

        # [CC] 发送继续信号 (0) 给所有 worker，通知它们准备参与推理
        # Signal workers to continue (0 = continue)
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        dist.broadcast(signal_tensor, src=0, group=self._signal_group)

        # [CC] 将转换后的观测广播给所有 worker 进程
        # Broadcast obs to workers
        self._broadcast_batch_to_workers(converted_obs)

        # [CC] 构建 Batch 对象供策略模型使用
        # Create batch for policy
        batch = Batch(obs=converted_obs)

        # [CC] 分布式前向推理：所有 rank 同步执行 barrier，然后并行前向传播
        # Distributed forward pass
        dist.barrier()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        dist.barrier()

        # [CC] 保存视频预测结果，后续可用于视频保存
        # Store video predictions for potential saving
        self.video_across_time.append(video_pred)

        # [CC] 从推理结果中提取动作，将 Batch 对象转为字典格式
        # Extract and convert action
        action_chunk_dict = result_batch.act

        # Convert Batch to dict
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)

        # [CC] 将 AR_droid 格式动作转换为 roboarena 的 (N, 8) 数组格式
        action = self._convert_action(action_dict)
        
        # Update first call flag
        if self._is_first_call:
            self._is_first_call = False
        
        return action
    
    # [CC] 内部状态重置方法：清空帧缓冲区、调用计数器等。
    # [CC] 可选在重置前将累积的视频预测解码保存为 MP4 文件。
    def _reset_state(self, save_video: bool = True) -> None:
        """Internal method to reset policy state.

        Args:
            save_video: Whether to save accumulated video before reset.
        """
        # [CC] 重置前可选地保存已累积的视频预测
        # Optionally save accumulated video before reset
        if save_video and len(self.video_across_time) > 0 and self._output_dir:
            try:
                frame_list = []
                # [CC] 将所有时间步的视频预测在时间维度上拼接
                video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                # [CC] 使用 VAE 解码器将潜在表示解码为像素帧，支持分块解码以节省显存
                frames = self._policy.trained_model.action_head.vae.decode(
                    video_across_time_cat,
                    tiled=self._policy.trained_model.action_head.tiled,
                    tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                    tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                )
                # [CC] 重排维度 B C T H W -> B T H W C，取第一个 batch，归一化到 [0,255] uint8
                frames = rearrange(frames, "B C T H W -> B T H W C")
                frames = frames[0]
                frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                for frame in frames:
                    frame_list.append(frame)

                if len(frame_list) > 0:
                    sample_frame = frame_list[0]
                    # [CC] 验证帧格式是否有效（H, W, C 且通道数为 1/3/4）
                    if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                        save_dir = self._output_dir
                        os.makedirs(save_dir, exist_ok=True)
                        # [CC] 按序号和时间戳命名保存 MP4 视频文件
                        all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                        num_frames = len(frame_list)
                        n = (num_frames - 1) // 8  # [CC] 计算自回归步数 n（帧数 = 8n+1）
                        output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                        imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                        logger.info(f"Saved video on reset to: {output_path}")
            except Exception as e:
                logger.warning(f"Failed to save video on reset: {e}")
        
        # [CC] 清空所有相机视角的帧缓冲区
        # Clear frame buffers
        for key in self._frame_buffers:
            self._frame_buffers[key] = []

        self._call_count = 0
        self._is_first_call = True
        self.video_across_time = []

    # [CC] 外部接口方法：重置策略状态，用于新 episode 开始时调用
    def reset(self, reset_info: dict) -> None:
        """Reset the policy state for a new episode.

        Clears frame buffers and resets call count.
        """
        self._reset_state(save_video=True)


# [CC] WebSocket 策略服务器类，使用 WebSocket 协议直接提供策略推理服务。
# [CC] 主要用于 rank 0 处理客户端连接、接收观测、执行推理并返回动作。
# [CC] 非 rank 0 进程通过 _worker_loop 参与分布式推理。
class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.
    Currently only implements the `load` and `infer` methods.
    """

    # [CC] 初始化 WebSocket 服务器，设置策略模型、网络参数、输出目录等
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        output_dir: str | None = None,
        signal_group: dist.ProcessGroup | None = None,
    ) -> None:
        self._policy = policy  # [CC] 策略模型实例
        self._host = host  # [CC] 服务器绑定地址
        self._port = port  # [CC] 服务器监听端口
        self._metadata = metadata or {}  # [CC] 发送给客户端的元数据（模型名称等）
        self._output_dir = output_dir  # [CC] 视频/输入数据保存目录
        logging.getLogger("websockets.server").setLevel(logging.INFO)
        self.video_across_time = []  # [CC] 跨时间步的视频预测累积列表
        self._msg_index = 0  # [CC] 消息序号
        self._signal_group = signal_group  # [CC] gloo 进程组，用于发送控制信号
        # [CC] 创建输出目录和输入数据子目录
        # Create output directory if specified
        if self._output_dir:
            os.makedirs(self._output_dir, exist_ok=True)
            os.makedirs(os.path.join(self._output_dir, "inputs"), exist_ok=True)
    
    # [CC] 保存每次收到的观测图像到磁盘，用于调试和数据记录。
    # [CC] 将 THWC 格式的多帧图像逐帧保存为 PNG 文件。
    def _save_input_obs(self, obs: dict) -> None:
        """Save incoming observation images per message.

        Expected format: THWC (Time, Height, Width, Channel) with 4 frames.
        Saves each frame as a separate PNG image: HWC format (uint8).

        Directory structure:
        output_dir/inputs/{msg_index:06d}_{timestamp}/{obs_key}/f{frame_idx:02d}.png
        """
        if not self._output_dir:
            return
        # [CC] 按消息序号和时间戳创建子目录
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        base_dir = os.path.join(self._output_dir, "inputs", f"{self._msg_index:06d}_{timestamp}")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception:
            return

        # [CC] 遍历三个相机视角的图像数据
        for key in ("video.exterior_image_1_left", "video.exterior_image_2_left", "video.wrist_image_left"):
            if key not in obs:
                continue
            value = obs[key]
            try:
                # [CC] 统一转为 numpy 数组
                # Convert to numpy if tensor
                if isinstance(value, torch.Tensor):
                    arr = value.detach().cpu().numpy()
                else:
                    arr = np.asarray(value)

                # [CC] 校验维度必须为 4D (T,H,W,C)
                # Expected format: THWC (Time, Height, Width, Channel)
                if arr.ndim != 4:
                    logger.warning(f"obs key '{key}' has shape {arr.shape}, expected 4D (T,H,W,C)")
                    continue

                # arr is (T, H, W, C)
                T, H, W, C = arr.shape

                # [CC] 将像素值归一化到 uint8 [0,255] 范围
                # Normalize to uint8
                if arr.dtype == np.uint8:
                    frames_u8 = arr
                else:
                    f = arr.astype(np.float32)
                    # [CC] 自动检测值域范围：[-1,1] 或其他
                    # Common conventions: [-1,1] or [0,1]
                    min_val = float(np.nanmin(f))
                    max_val = float(np.nanmax(f))
                    if min_val >= -1.1 and max_val <= 1.1:
                        # [CC] [-1,1] 范围，线性映射到 [0,255]
                        # Assume [-1,1] range
                        frames_u8 = ((f + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                    else:
                        # [CC] 其他范围，使用 min-max 归一化
                        # Min-max scaling
                        denom = (max_val - min_val) if (max_val - min_val) > 1e-6 else 1.0
                        frames_u8 = ((f - min_val) / denom * 255.0).clip(0, 255).astype(np.uint8)

                # [CC] 逐帧保存为 PNG 图像
                # Save each frame: frames_u8[i] is (H, W, C)
                key_dir = os.path.join(base_dir, key.replace("/", "_"))
                os.makedirs(key_dir, exist_ok=True)
                for frame_idx in range(T):
                    frame = frames_u8[frame_idx]  # (H, W, C)
                    # Handle grayscale (H, W) -> (H, W, 1)
                    if frame.ndim == 2:
                        frame = np.expand_dims(frame, axis=-1)
                    imageio.imwrite(os.path.join(key_dir, f"f{frame_idx:02d}.png"), frame)

            except Exception as e:
                logger.warning(f"Failed to save obs key '{key}': {e}")
                continue



    # [CC] 同步入口：启动异步事件循环运行服务器
    def serve_forever(self, rank: int = 0) -> None:
        asyncio.run(self.run(rank))

    # [CC] 异步主方法：rank 0 启动 WebSocket 服务器，其他 rank 启动 worker 循环
    async def run(self, rank: int = 0):
        if rank == 0:
            # [CC] rank 0 启动 WebSocket 服务，禁用压缩和消息大小限制以支持大图像数据
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
                ping_interval=None,
            ) as server:
                await server.serve_forever()
        else:
            # [CC] 非 rank 0 进程运行 worker 循环，等待并参与分布式推理
            # Non-rank-0 processes run a worker loop
            await self._worker_loop()

    # [CC] Worker 循环：非 rank 0 进程在此持续等待信号并参与分布式推理。
    # [CC] 信号协议：0=继续推理, 1=关闭退出, 2=空闲等待（客户端断连）
    async def _worker_loop(self):
        """Worker loop for non-rank-0 processes to participate in distributed inference."""
        logger.info(f"Worker loop started for rank {dist.get_rank()}")
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')
        while True:
            try:
                # [CC] 等待 rank 0 广播的控制信号
                # Wait for obs broadcast from rank 0
                # Create a dummy obs dict structure - will be filled by broadcast
                # obs = {}

                dist.broadcast(signal_tensor, src=0, group=self._signal_group)

                signal = signal_tensor.item()
                if signal == 1:
                    # [CC] 收到关闭信号，退出 worker 循环
                    logger.info(f"Rank {dist.get_rank()} received shutdown signal")
                    break

                # --- ADD THIS ELIF BLOCK ---
                elif signal == 2:
                    # [CC] 收到空闲信号（客户端断连），回到循环顶部等待下一个信号
                    logger.info(f"Rank {dist.get_rank()} received idle signal. Waiting for next client.")
                    # Loop back to the top and wait for the next signal
                    continue

                # [CC] 从 rank 0 接收广播的观测数据
                # Receive the batch data via broadcast/gather mechanism
                # This is a simplified version - the actual obs structure needs to be broadcasted
                batch = self._receive_batch_from_rank0()
                # [CC] 参与分布式前向推理（barrier 确保所有 rank 同步）
                # Participate in distributed forward pass
                dist.barrier()
                with torch.no_grad():
                    result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                dist.barrier()

            except Exception as e:
                logger.error(f"Worker loop error on rank {dist.get_rank()}: {e}")
                traceback.print_exc()
                break

    # [CC] 从 rank 0 接收广播的批数据：先接收大小，再接收数据，最后反序列化为 Batch 对象
    def _receive_batch_from_rank0(self):
        """Receive batch data from rank 0 using torch.distributed primitives."""
        import pickle

        # [CC] 先接收序列化数据的字节大小
        # Receive the size of the pickled data first
        size_tensor = torch.zeros(1, dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)
        data_size = size_tensor.item()

        # [CC] 根据大小分配缓冲区并接收实际数据
        # Receive the actual data
        data_tensor = torch.zeros(data_size, dtype=torch.uint8, device='cuda')
        dist.broadcast(data_tensor, src=0)

        # [CC] 反序列化：GPU tensor -> CPU bytes -> pickle 加载为字典 -> 包装为 Batch
        # Deserialize
        obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
        return Batch(obs=obs)

    # [CC] （WebsocketPolicyServer 版本）将观测数据从 rank 0 广播到所有 worker
    def _broadcast_batch_to_workers(self, obs):
        """Broadcast batch data from rank 0 to all other ranks."""
        import pickle

        # [CC] pickle 序列化观测字典
        # Serialize the obs
        serialized = pickle.dumps(obs)
        data_size = len(serialized)

        # [CC] 先广播数据大小
        # Broadcast size first
        size_tensor = torch.tensor([data_size], dtype=torch.int64, device='cuda')
        dist.broadcast(size_tensor, src=0)

        # [CC] 再广播数据本体
        # Broadcast data
        data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
        dist.broadcast(data_tensor, src=0)

    # [CC] WebSocket 连接处理器：处理单个客户端连接的完整生命周期。
    # [CC] 流程：发送元数据 -> 循环接收观测 -> 分布式推理 -> 返回动作 -> 定期保存视频
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        # [CC] 连接建立后首先发送模型元数据给客户端
        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        signal_tensor = torch.zeros(1, dtype=torch.int32, device='cpu')

        try:
            while True:
                try:
                    # [CC] 等待并接收客户端发送的观测数据
                    start_time = time.perf_counter()
                    data = await websocket.recv()
                    recv_done = time.perf_counter()
                    # [CC] 使用 msgpack 反序列化观测数据（支持 numpy 数组）
                    obs = msgpack_numpy.unpackb(data)
                    print(f"Wait Time: {recv_done - start_time:.2f} seconds")
                    self._msg_index += 1

                    infer_start_time = time.perf_counter()

                    # [CC] 通过 gloo 进程组发送继续信号 (0) 给所有 worker
                    # Signal other ranks to continue (0 = continue)
                    signal_tensor.zero_()
                    dist.broadcast(signal_tensor, src=0, group=self._signal_group) # <-- USE GLOO GROUP

                    # [CC] 将观测数据广播给所有 rank 用于分布式推理
                    # Broadcast the obs to all ranks for distributed inference
                    self._broadcast_batch_to_workers(obs)
                    batch = Batch(obs=obs)

                    # [CC] 所有 rank 通过 barrier 同步后执行前向推理
                    # All ranks need to participate in the forward pass
                    dist.barrier()
                    forward_start_time = time.perf_counter()
                    with torch.no_grad():
                        result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
                    dist.barrier()
                    print(f"Forward Time: {time.perf_counter() - forward_start_time:.2f} seconds")

                    # [CC] 提取推理结果中的动作和视频预测
                    action_chunk_dict = result_batch.act
                    video_chunk = video_pred

                    print(f"Inference Time: {time.perf_counter() - infer_start_time:.2f} seconds")

                    # [CC] 累积视频预测用于后续保存
                    self.video_across_time.append(video_chunk)

                    # [CC] 当累积超过 10 个视频块时，解码并保存为 MP4 文件
                    if len(self.video_across_time) > 10:
                        frame_list = []
                        # [CC] 在时间维度拼接所有累积的视频潜在表示
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        # [CC] VAE 解码：潜在空间 -> 像素空间，使用分块策略减少显存占用
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        # [CC] 重排维度并归一化到 [0,255] uint8
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # [CC] 按序号和时间戳命名，保存为 MP4 视频
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # [CC] 帧数 = 8n+1，反推自回归步数 n
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                        # [CC] 保存后清空视频累积列表
                        self.video_across_time = []
                    # [CC] 特殊情况：当自回归起始帧恰好为一个块之后，且有多段视频时，
                    # [CC] 保存除最后一段外的所有视频（最后一段是新块的开始，需要保留）
                    elif self._policy.trained_model.action_head.current_start_frame == 1 + self._policy.trained_model.action_head.num_frame_per_block and len(self.video_across_time) > 1:
                        print("current_start_frame == 1 + num_frame_per_block and len(self.video_across_time) > 1")
                        frame_list = []
                        # [CC] 拼接除最后一段外的所有视频预测，保留最后一段作为新块的起始
                        video_across_time_cat = torch.cat(self.video_across_time[:-1], dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        # [CC] 保留最后一段视频作为下一轮的起始
                        self.video_across_time = [video_chunk]


                    # [CC] 将 Batch 对象转换为字典，只保留以 "action." 开头的键
                    def batch_to_dict(batch):
                        out = {}
                        for k in dir(batch):
                            if not k.startswith("action."):
                                continue
                            out[k] = getattr(batch, k)
                        return out
                    action_chunk_dict = batch_to_dict(action_chunk_dict)
                    # [CC] 将动作字典通过 msgpack 序列化后发送给客户端
                    await websocket.send(packer.pack(action_chunk_dict))

                except websockets.ConnectionClosed:
                    # [CC] 客户端断开连接时，保存所有累积的视频预测
                    logger.info(f"Connection from {websocket.remote_address} closed")
                    if len(self.video_across_time) > 0:
                        frame_list = []
                        video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                        frames = self._policy.trained_model.action_head.vae.decode(
                            video_across_time_cat,
                            tiled=self._policy.trained_model.action_head.tiled,
                            tile_size=(self._policy.trained_model.action_head.tile_size_height, self._policy.trained_model.action_head.tile_size_width),
                            tile_stride=(self._policy.trained_model.action_head.tile_stride_height, self._policy.trained_model.action_head.tile_stride_width),
                        )
                        frames = rearrange(frames, "B C T H W -> B T H W C")
                        frames = frames[0]
                        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                        # Add each frame individually to the list
                        for frame in frames:
                            frame_list.append(frame)

                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            # Save all frames as a single MP4 file
                            save_dir = self._output_dir if self._output_dir else "."
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8  # num_frames = 8n+1, so n = (num_frames-1)/8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video to: {output_path}")
                        else:
                            print(f"Warning: Invalid frame shape {sample_frame.shape}. Expected (H, W, C) with C in [1, 3, 4]. Skipping video save.")

                    # [CC] 清空视频累积列表并退出消息循环
                    self.video_across_time = []
                    break
                except Exception:
                    # [CC] 发生异常时将错误堆栈发送给客户端，然后关闭连接
                    await websocket.send(traceback.format_exc())
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                    raise
        finally:
            # [CC] 连接结束后发送空闲信号 (2) 给所有 worker，让它们回到等待状态
            logger.info(f"Rank 0: Client session ended. Sending idle signal (2) to workers.")
            signal_tensor.fill_(2)  # Set tensor value to 2
            dist.broadcast(signal_tensor, src=0, group=self._signal_group)
            # When connection closes, signal other ranks to continue waiting for next connection
            # (or implement proper shutdown if needed)


# [CC] 初始化分布式设备网格：启动 NCCL 进程组，为每个 rank 分配 GPU 设备
def init_mesh() -> DeviceMesh:
    # [CC] 初始化 NCCL 后端的分布式进程组（环境变量由 torchrun 设置）
    # env vars set by torchrun
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) setting device to {rank}")

    # [CC] 每个 rank 绑定对应编号的 CUDA 设备
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # [CC] 创建一维设备网格，维度名为 "ip"（推理并行）
    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size, ),
        mesh_dim_names=("ip", ),
    )
    print(f"Rank {rank}/{world_size} (PID: {os.getpid()}) using device {device}")

    return mesh

# [CC] HTTP 健康检查端点：响应 /healthz 路径返回 200 OK，用于服务存活探测
def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # [CC] 非健康检查请求则交由正常的 WebSocket 处理流程
    # Continue with the normal request handling.
    return None


# [CC] 主函数：初始化环境、加载模型、启动分布式推理服务器
def main(args: Args) -> None:
    # [CC] 设置 DiT 缓存环境变量
    # Set environment variable for DIT cache.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"

    # [CC] 使用 Transformer Engine 的 cuDNN 后端进行注意力计算
    # Use TE cuDNN backend for attention.
    os.environ["ATTENTION_BACKEND"] = "TE"

    # [CC] 提高 torch.compile 的重编译次数限制（自回归模型有多种可能的张量形状）
    # Increase the recompile limit to 100 for inference due
    # to autoregressive nature of the model (several possible shapes).
    torch._dynamo.config.recompile_limit = 800

    # [CC] 设定机器人类型和模型路径
    embodiment_tag = "oxe_droid"
    model_path = args.model_path
    policy_metadata = {
        "embodiment": embodiment_tag,
        "model_name": "dreamzero",
        "model_path": model_path,
    }

    # [CC] 初始化分布式设备网格和 NCCL 进程组
    device_mesh = init_mesh()
    rank = dist.get_rank()

    # [CC] 创建 gloo 后端的信号进程组，用于 CPU 上的控制信号广播
    # [CC] gloo 后端在 CPU 上运行，比 NCCL 更适合传输小的控制信号
    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)
    logger.info(f"Rank {rank} initialized signal_group (gloo)")

    # [CC] 加载 GrootSimPolicy 模型，支持分布式推理
    # 加载时根据训练参数判断是否是lora, 无需额外指定
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag(embodiment_tag),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
    )

    # [CC] rank 0 负责 WebSocket 服务和视频保存，其他 rank 作为 worker
    # Create server for all ranks - rank 0 handles websocket, others run worker loop
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
        # [CC] 根据模型路径和日期创建视频输出目录
        # Create output directory for videos
        # Extract parent directory and checkpoint name from model_path
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(parent_dir, f"real_world_eval_gen_{date_suffix}_{args.index}", checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Videos will be saved to: %s", output_dir)
    else:
        output_dir = None
        logging.info(f"Rank {rank} starting as worker for distributed inference...")

    # [CC] 创建策略包装器，负责 roboarena 和 AR_droid 格式之间的转换
    # Create wrapper policy that converts between roboarena and AR_droid formats
    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    # [CC] 配置 AR_droid 服务器参数：2个外部相机、腕部相机、关节位置动作空间
    # Configure server for AR_droid (2 external cameras, wrist camera, joint position actions)
    server_config = PolicyServerConfig(
        image_resolution=(180, 320),  # [CC] AR_droid 期望 180x320 分辨率的图像
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # [CC] 启用会话跟踪，新客户端连接时重置状态
        action_space="joint_position",
    )

    if rank == 0:
        # [CC] rank 0 启动 roboarena WebSocket 策略服务器，直接处理客户端请求
        logging.info("Using roboarena policy server interface")
        logging.info(f"Server config: {server_config}")
        roboarena_server = RoboarenaServer(
            policy=wrapper_policy,
            server_config=server_config,
            host="0.0.0.0",
            port=args.port,
        )
        roboarena_server.serve_forever()
    else:
        # [CC] 非 rank 0 进程创建 WebsocketPolicyServer 并运行 worker 循环
        # [CC] worker 循环会等待 rank 0 的信号并参与分布式前向推理
        # Non-rank-0 processes need to run worker loop for distributed inference
        # We'll use the existing WebsocketPolicyServer's worker loop mechanism
        server = WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata=policy_metadata,
            output_dir=output_dir,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())
    


# [CC] 程序入口：配置日志、解析命令行参数、启动主函数
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)  # [CC] 使用 tyro 自动从 Args 数据类生成命令行参数解析器
    main(args)