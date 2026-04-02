"""Server for serving a policy over websockets.

Adapted from https://github.com/robo-arena/roboarena/

"""

# [CC] 导入异步IO、数据类、日志和异常追踪等标准库模块
import asyncio
import dataclasses
import logging
import traceback

# [CC] 导入策略基类、消息序列化工具和WebSocket服务器相关模块
from openpi_client.base_policy import BasePolicy
from openpi_client import msgpack_numpy
import websockets.asyncio.server
import websockets.frames


# [CC] 策略服务器配置数据类，定义了客户端与服务器之间通信所需的各项参数，
# [CC] 包括图像分辨率、相机配置、会话ID需求以及动作空间类型等。
@dataclasses.dataclass
class PolicyServerConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    # It's beneficial to resize images to the desired resolution client-side for faster communication.
    # [CC] 客户端图像缩放分辨率，None表示不缩放；在客户端缩放可减少传输数据量
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    # [CC] 是否需要发送腕部摄像头图像
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    # [CC] 需要发送的外部摄像头数量，取值范围为 [0, 1, 2]
    n_external_cameras: int = 1  # can be in [0, 1, 2]
    # Whether or not stereo camera image(s) should be sent.
    # [CC] 是否需要发送立体摄像头（双目）图像
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent (e.g. for policies that want to keep track of history).
    # [CC] 是否需要发送唯一的评估会话ID（用于需要追踪历史记录的策略）
    needs_session_id: bool = False
    # Which action space to use.
    # [CC] 动作空间类型，可选：关节位置、关节速度、笛卡尔位置、笛卡尔速度
    action_space: str = "joint_position"  # can be in ["joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity"]


# [CC] 基于WebSocket协议的策略服务器类。
# [CC] 负责通过WebSocket接收客户端（机器人端）发来的观测数据，调用策略模型进行推理，
# [CC] 然后将预测的动作序列返回给客户端执行。支持reset和infer两种端点。
class WebsocketPolicyServer:
    """
    Serves a policy using the websocket protocol.

    Interface:
      Observation:
        - observation/wrist_image_left: (H, W, 3) if needs_wrist_camera is True
        - observation/wrist_image_right: (H, W, 3) if needs_wrist_camera is True and needs_stereo_camera is True
        - observation/exterior_image_{i}_left: (H, W, 3) if n_external_cameras >= 1
        - observation/exterior_image_{i}_right: (H, W, 3) if needs_stereo_camera is True
        - session_id: (1,) if needs_session_id is True
        - observation/joint_position: (7,)
        - observation/cartesian_position: (6,)
        - observation/gripper_position: (1,)
        - prompt: str, the natural language task instruction for the policy

      Action:
        - action: (N, 8,) or (N, 7,): either 7 movement actions (for joint action spaces) or 6 (for cartesian) plus one dimension for gripper position
                           --> all N actions will get executed on the robot before the server is queried again

    """

    # [CC] 初始化策略服务器，接收策略对象、服务器配置、监听地址和端口
    def __init__(
        self,
        policy: BasePolicy,
        server_config: PolicyServerConfig,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        self._policy = policy
        self._server_config = server_config
        self._host = host
        self._port = port
        # [CC] 设置WebSocket服务器日志级别为INFO，避免过多调试信息
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    # [CC] 阻塞式启动服务器，内部通过asyncio事件循环运行异步服务
    def serve_forever(self) -> None:
        asyncio.run(self.run())

    # [CC] 异步启动WebSocket服务器，绑定到指定地址和端口，持续监听连接
    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,  # [CC] 禁用压缩，减少延迟
            max_size=None,  # [CC] 不限制消息大小，因为图像数据可能很大
        ) as server:
            await server.serve_forever()

    # [CC] WebSocket连接处理函数，每当有新客户端连接时被调用。
    # [CC] 负责：1) 发送服务器配置给客户端；2) 循环接收观测数据并返回动作或重置确认。
    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        # [CC] 创建msgpack序列化器，用于将数据打包为二进制格式（支持numpy数组）
        packer = msgpack_numpy.Packer()

        # Send server config to client to configure what gets sent to server.
        # [CC] 首次连接时将服务器配置发送给客户端，客户端据此决定发送哪些观测数据
        await websocket.send(packer.pack(dataclasses.asdict(self._server_config)))

        # [CC] 主循环：持续接收客户端请求并返回响应
        while True:
            try:
                # [CC] 接收并反序列化客户端发来的观测数据
                obs = msgpack_numpy.unpackb(await websocket.recv())

                # [CC] 提取请求端点类型（"reset" 或 "infer"），并从观测字典中移除该字段
                endpoint = obs["endpoint"]
                del obs["endpoint"]
                if endpoint == "reset":
                    # [CC] 重置端点：重置策略状态（如清除历史缓存），返回成功消息
                    self._policy.reset(obs)
                    to_return = "reset successful"
                else:
                    # [CC] 推理端点：调用策略模型进行推理，返回序列化后的动作序列
                    action = self._policy.infer(obs)
                    to_return = packer.pack(action)
                await websocket.send(to_return)
            except websockets.ConnectionClosed:
                # [CC] 客户端断开连接，正常退出循环
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                # [CC] 发生未预期的异常：将异常堆栈信息发送给客户端，然后关闭连接并抛出异常
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


# [CC] 主程序入口：使用虚拟策略（输出全零动作）启动服务器，用于测试和调试
if __name__ == "__main__":
    import numpy as np

    # [CC] 虚拟策略类，继承BasePolicy，用于测试服务器功能
    # [CC] infer方法返回全零动作，reset方法不做任何操作
    class DummyPolicy(BasePolicy):
        # [CC] 推理方法：返回形状为(1, 8)的全零动作（1步，8维：7维运动+1维夹爪）
        def infer(self, obs):
            return np.zeros((1, 8), dtype=np.float32)

        # [CC] 重置方法：空实现，虚拟策略无需重置状态
        def reset(self, reset_info):
            pass

    # [CC] 配置日志并启动策略服务器
    logging.basicConfig(level=logging.INFO)
    policy = DummyPolicy()
    server = WebsocketPolicyServer(policy, PolicyServerConfig())
    server.serve_forever()
        