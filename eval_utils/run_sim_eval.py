"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control),
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

# [CC] 导入标准库和第三方依赖：uuid用于生成唯一会话ID，tyro/argparse用于命令行参数解析，
# gymnasium用于仿真环境接口，torch/cv2/mediapy/numpy用于张量操作、图像处理和视频录制

import uuid

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# [CC] openpi_client提供图像预处理工具（如resize_with_pad），
# InferenceClient是推理客户端的抽象基类，WebsocketClientPolicy通过WebSocket与策略服务器通信
from openpi_client import image_tools
from sim_evals.inference.abstract_client import InferenceClient
from policy_client import WebsocketClientPolicy


# [CC] DreamZero关节位置控制推理客户端，继承自InferenceClient抽象基类。
# [CC] 负责与远程策略服务器通信，发送观测数据并接收动作预测结果。
# [CC] 支持开环（open-loop）动作块执行：一次推理获取多步动作，按步依次执行，减少推理频率。
class DreamZeroJointPosClient(InferenceClient):
    # [CC] 初始化推理客户端，建立与远程策略服务器的WebSocket连接
    # [CC] remote_host/remote_port: 策略服务器地址和端口
    # [CC] open_loop_horizon: 开环执行步数，即每次推理后连续执行多少步动作再重新推理
    def __init__(self,
                remote_host:str = "localhost",
                remote_port:int = 6000,
                open_loop_horizon:int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(remote_host, remote_port)
        self.open_loop_horizon = open_loop_horizon
        # [CC] 记录当前动作块中已执行的步数
        self.actions_from_chunk_completed = 0
        # [CC] 缓存策略服务器返回的动作块（多步动作序列）
        self.pred_action_chunk = None
        # [CC] 用UUID标识当前推理会话，便于服务端区分不同episode
        self.session_id = str(uuid.uuid4())

    # [CC] 可视化方法：将三个摄像头图像（右侧、手腕、左侧）调整为224x224并水平拼接，
    # [CC] 用于展示模型实际看到的视角
    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        # [CC] 对每个摄像头图像进行等比缩放并填充到224x224
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        # [CC] 水平拼接三个视角的图像，形成宽度为672的全景图
        combined = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return combined

    # [CC] 重置客户端状态，在每个episode结束后调用。
    # [CC] 清空动作缓存并生成新的会话ID，确保下一个episode从头开始推理
    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    # [CC] 核心推理方法：根据当前观测和语言指令，返回下一步动作及可视化图像。
    # [CC] 采用开环动作块策略：当动作块用完或首次调用时，向服务器请求新的动作块；
    # [CC] 否则直接从缓存的动作块中取出下一步动作，避免每步都调用模型推理。
    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        curr_obs = self._extract_observation(obs)
        # [CC] 判断是否需要重新向服务器请求新的动作块：
        # [CC] 1) 首次调用（completed==0且chunk为None）；2) 当前动作块已执行完毕
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            # [CC] 构建请求数据：包含三个摄像头图像（缩放到180x320）、关节位置、夹爪位置、语言指令和会话ID
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                # [CC] 笛卡尔位置用零向量占位，当前模型不依赖此输入
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),  # dummy cartesian position
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }
            # [CC] 打印请求数据的形状信息，用于调试
            for k, v in request_data.items():
                print(f"{k}: {v.shape if not isinstance(v, str) else v}")

            # [CC] 通过WebSocket向策略服务器发送推理请求，获取动作块
            result = self.client.infer(request_data)
            actions = result["actions"] if isinstance(result, dict) else result
            # [CC] 校验返回的动作形状：必须是2D数组，最后一维为8（7个关节 + 1个夹爪）
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            assert actions.shape[-1] == 8, f"Expected 8 action dimensions (7 joints + 1 gripper), got {actions.shape[-1]}"
            self.pred_action_chunk = actions

        # [CC] 从动作块中取出当前步对应的动作
        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # [CC] 将夹爪动作二值化：大于0.5为完全打开（1），否则为完全关闭（0）
        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        # [CC] 生成可视化图像：将三个摄像头视角缩放到224x224并水平拼接
        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        both = np.concatenate([img1, img2, img3], axis=1)

        return {"action": action, "viz": both}

    # [CC] 从仿真环境的原始观测字典中提取并整理所需的观测数据。
    # [CC] 将GPU上的PyTorch张量转换为CPU上的NumPy数组，供推理客户端使用。
    # [CC] 可选地将摄像头图像保存到磁盘用于调试。
    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # [CC] 从观测字典中提取三个摄像头的图像，索引[0]取batch中的第一个环境
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # [CC] 提取本体感知状态：7维关节角度和夹爪开合位置
        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        # [CC] 调试用：将右侧和手腕摄像头图像拼接后保存为PNG
        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }




# [CC] 主函数：在IsaacLab仿真环境中运行DreamZero策略的评估。
# [CC] 参数说明：
# [CC]   episodes: 评估的episode数量
# [CC]   scene: 场景编号（1=方块放碗里, 2=罐子放杯子里, 3=香蕉放垃圾桶里）
# [CC]   headless: 是否无头模式运行（不显示GUI窗口）
# [CC]   host/port: 策略服务器的地址和端口
def main(
        episodes: int = 10,
        scene: int = 1,
        headless: bool = True,
        host: str = "localhost",
        port: int = 6000,
        ):
    # [CC] 在函数内部启动Omniverse应用，避免与tyro命令行解析冲突
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    # [CC] 启用摄像头渲染，并根据参数决定是否使用无头模式
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # [CC] IsaacLab相关模块必须在应用启动后才能导入，否则会报错
    # All IsaacLab dependent modules should be imported after the app is launched
    import sim_evals.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg


    # [CC] 解析DROID环境配置：指定设备、环境数量为1、启用Fabric高性能后端
    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    # [CC] 根据场景编号选择对应的语言指令
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "pick up the can and put it in the mug"
        case 3:
            instruction = "put the banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")

    # [CC] 设置场景并创建Gymnasium环境实例
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    # [CC] 重置两次环境：第二次重置是为了确保材质/纹理在渲染管线中正确加载
    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    # [CC] 创建推理客户端，连接到策略服务器
    client = DreamZeroJointPosClient(remote_host=host, remote_port=port)

    # [CC] 创建视频保存目录，按日期和时间组织
    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    # [CC] 评估循环：关闭梯度计算以节省显存
    with torch.no_grad():
        for ep in range(episodes):
            # [CC] 每个episode执行最多max_steps步
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                # [CC] 调用推理客户端获取动作和可视化图像
                ret = client.infer(obs, instruction)
                # [CC] 非无头模式下，用OpenCV窗口实时显示摄像头画面
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                # [CC] 收集视频帧
                video.append(ret["viz"])
                # [CC] 将动作转为PyTorch张量并增加batch维度，然后执行环境步进
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                # [CC] 如果episode终止（成功或超时），提前退出内层循环
                if term or trunc:
                    break

            # [CC] 每个episode结束后重置客户端状态，并将收集的帧保存为MP4视频
            client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    # [CC] 关闭仿真环境和Omniverse应用，释放资源
    env.close()
    simulation_app.close()

# [CC] 入口点：使用tyro库自动将main函数的参数暴露为命令行接口
if __name__ == "__main__":
    args = tyro.cli(main)
