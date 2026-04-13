#!/usr/bin/env python3
"""
Block Influence (BI) 分析 —— DreamZero 版

在真实推理中，收集 CausalWanAttentionBlock 的 BI 数据。
由于 DreamZero 将 video 和 action token 拼接在同一序列中，
因此可以分别计算 video 区域和 action 区域的 BI。

数据维度：
    records[region][block_idx][timestep_idx] = [bi_chunk0, bi_chunk1, ...]
    region: 'video' / 'action' / 'all'

启动方式（替换正常 server 启动）：
    torchrun --nproc_per_node <N> analysis/block_bi_analysis.py \
        --model-path ./checkpoints/dreamzero \
        --port 8000 \
        --bi-save-dir analysis/bi_results

然后正常启动评测 client 跑推理。Ctrl+C 或 client 断连后自动保存 BI 统计和可视化。
"""

import argparse
import logging
import os
import sys
import datetime
import socket
import asyncio

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'Droid Sans Fallback'
matplotlib.rcParams['axes.unicode_minus'] = False
from collections import defaultdict

# ---------------------------------------------------------------------------
# 添加项目路径
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BICollector：通过 hook 收集 BI 数据
# ---------------------------------------------------------------------------

class BICollector:
    """
    通过 forward hook 收集每个 CausalWanAttentionBlock 的 Block Influence。

    设计思路：
      1. 在每个 block 上注册 forward hook，计算 BI = 1 - cos_sim(h_in, h_out)。
         由于 video 和 action token 拼接在 x 的同一序列中，hook 会分别计算：
         - video 区域: x[:, :seq_len, :]
         - action 区域: x[:, seq_len:seq_len+action_len, :]
         - 整体: 全部 token

      2. monkey patch CausalWanModel._forward_blocks 以追踪：
         - seq_len（video 区域长度）
         - action_length（action 区域长度）
         - timestep 计数器

      3. monkey patch WANPolicyHead.lazy_joint_video_action 以追踪 chunk 边界
         并在每个新 chunk 开始时重置 timestep 计数器。

    数据结构：
        records[region][block_idx][timestep_idx] = [bi_chunk0, bi_chunk1, ...]
    """

    def __init__(self, action_head):
        """
        Args:
            action_head: WANPolicyHead 实例（policy.trained_model.action_head）
        """
        self.action_head = action_head
        self.model = action_head.model  # CausalWanModel
        self.num_blocks = len(self.model.blocks)

        # 当前推理状态
        self._current_chunk_id = -1
        self._timestep_counter = 0
        self._current_seq_len = 0       # video token 长度
        self._current_action_len = 0    # action token 长度

        # 数据容器: records[region][block_idx][timestep_idx] = [bi values across chunks]
        self.records = {
            'video': defaultdict(lambda: defaultdict(list)),
            'action': defaultdict(lambda: defaultdict(list)),
        }

        self._hooks = []
        self._install_hooks()

    def _install_hooks(self):
        # 1. 每个 block 的 forward hook：计算并记录 BI
        for idx in range(self.num_blocks):
            def make_hook(block_idx):
                def hook_fn(module, inputs, output, **kwargs):
                    # block 被以关键字参数调用: block(x=x, e=e0, ...)
                    # 所以 inputs 可能为空，x 在 kwargs 中
                    if inputs:
                        h_in = inputs[0]
                    else:
                        h_in = kwargs.get('x', None)
                        if h_in is None:
                            return
                    h_out = output[0] if isinstance(output, tuple) else output  # [B, L, C]

                    seq_len = self._current_seq_len
                    action_len = self._current_action_len

                    # Video 区域 BI
                    if seq_len > 0 and h_in.shape[1] >= seq_len:
                        cos_video = F.cosine_similarity(
                            h_in[:, :seq_len].float(),
                            h_out[:, :seq_len].float(), dim=-1)
                        bi_video = 1.0 - cos_video.mean().item()
                        self.records['video'][block_idx][
                            self._timestep_counter].append(bi_video)

                    # Action 区域 BI
                    if action_len > 0 and h_in.shape[1] >= seq_len + action_len:
                        cos_action = F.cosine_similarity(
                            h_in[:, seq_len:seq_len + action_len].float(),
                            h_out[:, seq_len:seq_len + action_len].float(),
                            dim=-1)
                        bi_action = 1.0 - cos_action.mean().item()
                        self.records['action'][block_idx][
                            self._timestep_counter].append(bi_action)

                return hook_fn
            h = self.model.blocks[idx].register_forward_hook(make_hook(idx), with_kwargs=True)
            self._hooks.append(h)

        # 2. monkey patch _forward_blocks 以追踪 seq_len, action_length 和 timestep
        original_forward_blocks = self.model._forward_blocks

        def wrapped_forward_blocks(*args, **kwargs):
            # _forward_blocks 可能以关键字参数调用: _forward_blocks(x=x, seq_len=seq_len, ...)
            seq_len = kwargs.get('seq_len', args[1] if len(args) > 1 else 0)
            self._current_seq_len = seq_len

            action = kwargs.get('action', args[7] if len(args) > 7 else None)
            if action is not None:
                self._current_action_len = action.shape[1]
            else:
                self._current_action_len = 0

            result = original_forward_blocks(*args, **kwargs)

            # 每次 _forward_blocks 完成后递增 timestep
            self._timestep_counter += 1

            return result

        self.model._forward_blocks = wrapped_forward_blocks

        # 3. monkey patch lazy_joint_video_action 以追踪 chunk 边界
        original_lazy = self.action_head.lazy_joint_video_action

        def wrapped_lazy(*args, **kwargs):
            self._current_chunk_id += 1
            self._timestep_counter = 0
            result = original_lazy(*args, **kwargs)
            return result

        self.action_head.lazy_joint_video_action = wrapped_lazy

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def save_and_plot(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)

        for region in ['video', 'action']:
            records = self.records[region]
            if not records[0]:
                logger.info(f'[BI] {region} 无数据，跳过')
                continue

            # 找出最大 timestep 数
            max_t = max(max(records[b].keys()) for b in range(self.num_blocks)
                        if records[b]) + 1
            num_chunks = len(records[0][0])

            # 构建矩阵: [num_blocks, max_t, num_chunks]
            bi_cube = np.zeros((self.num_blocks, max_t, num_chunks))
            for b in range(self.num_blocks):
                for t in range(max_t):
                    vals = records[b].get(t, [])
                    for c, v in enumerate(vals):
                        if c < num_chunks:
                            bi_cube[b, t, c] = v

            # 对 chunk 维度取均值和标准差
            bi_mean_bt = bi_cube.mean(axis=2)  # [N_blocks, T]
            bi_std_bt = bi_cube.std(axis=2)

            np.save(os.path.join(save_dir, f'bi_cube_{region}.npy'), bi_cube)
            np.save(os.path.join(save_dir, f'bi_mean_bt_{region}.npy'), bi_mean_bt)
            np.save(os.path.join(save_dir, f'bi_std_bt_{region}.npy'), bi_std_bt)

            # 对 timestep 维度聚合
            bi_mean_b = bi_mean_bt.mean(axis=1)  # [N_blocks]
            bi_std_b = bi_mean_bt.std(axis=1)

            # --- 图1: 热力图 ---
            fig, ax = plt.subplots(figsize=(max(12, max_t // 2), 8))
            im = ax.imshow(bi_mean_bt, aspect='auto', cmap='hot', origin='upper')
            ax.set_xlabel('Timestep index (0=noise, T=clean)')
            ax.set_ylabel(f'Block index (0=first, {self.num_blocks-1}=last)')
            ax.set_title(f'[{region}] Block Influence 随时间步分布\n'
                         f'颜色越亮=该 block 在该时间步对 {region} token 影响越大 '
                         f'({num_chunks} chunks 均值)')
            ax.set_yticks(range(0, self.num_blocks, 2))
            plt.colorbar(im, ax=ax, label='BI = 1 - cosine_sim(input, output)')
            plt.tight_layout()
            plt.savefig(os.path.join(
                save_dir,
                f'{region}去噪过程中每个block在每个时间步的BI均值.png'), dpi=150)
            plt.close()

            # --- 图2: 散点图 ---
            fig, ax = plt.subplots(figsize=(8, 7))
            sc = ax.scatter(bi_mean_b, bi_std_b,
                            c=np.arange(self.num_blocks),
                            cmap='tab20', s=80, zorder=3)
            for i, (mx, sx) in enumerate(zip(bi_mean_b, bi_std_b)):
                ax.annotate(str(i), (mx, sx), textcoords='offset points',
                            xytext=(5, 3), fontsize=7)
            ax.axvline(np.median(bi_mean_b), color='gray', linestyle='--',
                       linewidth=0.8, alpha=0.6, label='均值中位数')
            ax.axhline(np.median(bi_std_b), color='gray', linestyle='--',
                       linewidth=0.8, alpha=0.6, label='方差中位数')
            ax.set_xlabel('BI均值（越低说明该block越冗余）')
            ax.set_ylabel('BI跨时间步标准差（越低说明重要性越稳定）')
            ax.set_title(f'{region}去噪中各block的冗余程度\n'
                         f'左下=全程可跳, 左上=部分时间步可跳, 右侧=不可跳')
            ax.legend(fontsize=8)
            plt.colorbar(sc, ax=ax, label='Block编号')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(
                save_dir,
                f'{region}去噪中各block的BI均值与跨时间步方差散点图.png'), dpi=150)
            plt.close()

            # 打印最冗余 block
            top5 = np.argsort(bi_mean_b)[:5]
            logger.info(f'\n[{region}] 最冗余的 5 个 block（BI 均值最低）:')
            for idx in top5:
                logger.info(
                    f'  Block {idx:2d}: mean={bi_mean_b[idx]:.4f}, '
                    f'timestep_std={bi_std_b[idx]:.4f}')

        # --- 图3: video vs action 对比 ---
        video_mean = np.zeros(self.num_blocks)
        action_mean = np.zeros(self.num_blocks)
        for b in range(self.num_blocks):
            v_vals = [v for t_vals in self.records['video'][b].values()
                      for v in t_vals]
            a_vals = [v for t_vals in self.records['action'][b].values()
                      for v in t_vals]
            if v_vals:
                video_mean[b] = np.mean(v_vals)
            if a_vals:
                action_mean[b] = np.mean(a_vals)

        if video_mean.any() and action_mean.any():
            diff = action_mean - video_mean
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
            x = np.arange(self.num_blocks)

            ax1.plot(x, video_mean, 'o-', label='video区域',
                     color='steelblue', linewidth=1.5, markersize=4)
            ax1.plot(x, action_mean, 's-', label='action区域',
                     color='tomato', linewidth=1.5, markersize=4)
            ax1.set_ylabel('BI均值')
            ax1.set_title('各block在video区域和action区域的BI均值对比')
            ax1.legend()
            ax1.set_xticks(x[::2])
            ax1.grid(True, alpha=0.3)

            colors = ['tomato' if d > 0 else 'steelblue' for d in diff]
            ax2.bar(x, diff, color=colors, alpha=0.8)
            ax2.axhline(0, color='black', linewidth=0.8)
            ax2.set_xlabel('Block编号')
            ax2.set_ylabel('BI_action - BI_video')
            ax2.set_title('各block的区域偏向（正值=对action更重要，负值=对video更重要）')
            ax2.set_xticks(x[::2])
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(
                save_dir,
                '各block在video和action区域的BI对比及区域偏向.png'), dpi=150)
            plt.close()

        logger.info(f'\n[BI] 结果已保存至: {save_dir}')


# ---------------------------------------------------------------------------
# 入口：复用 socket_test_optimized_AR.py 的启动逻辑，注入 BI hook
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DreamZero Block Influence 分析')
    parser.add_argument("--model-path", type=str, default="./checkpoints/dreamzero",
                        help='模型检查点路径')
    parser.add_argument("--port", type=int, default=8000,
                        help='WebSocket 服务器端口')
    parser.add_argument("--bi-save-dir", type=str, default="analysis/bi_results",
                        help='BI 分析结果保存目录')
    parser.add_argument("--enable-dit-cache", action="store_true",
                        help='是否启用 DiT 缓存')
    parser.add_argument("--timeout-seconds", type=int, default=50000,
                        help='分布式通信超时时间')
    parser.add_argument("--index", type=int, default=0,
                        help='实验索引')
    parser.add_argument("--max-chunk-size", type=int, default=None,
                        help='推理时最大 chunk 大小')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)

    # --- 环境设置（同 socket_test_optimized_AR.py）---
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    # --- 分布式初始化 ---
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(world_size,),
        mesh_dim_names=("ip",),
    )

    timeout_delta = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout_delta)

    # --- 加载模型 ---
    model_path = args.model_path
    model_config_overrides = []
    if args.max_chunk_size is not None:
        model_config_overrides.append(
            f"action_head_cfg.config.diffusion_model_cfg.max_chunk_size={args.max_chunk_size}")

    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=mesh,
        model_config_overrides=model_config_overrides if model_config_overrides else None,
    )

    # --- [BI] 注入 hook（仅 rank 0 收集数据）---
    collector = None
    if rank == 0:
        action_head = policy.trained_model.action_head
        collector = BICollector(action_head)
        logger.info(f'[BI] Hook 已注册到 {collector.num_blocks} 个 block，'
                    f'数据将保存至 {args.bi_save_dir}')

    # --- 启动 server（复用 socket_test_optimized_AR 的逻辑）---
    from socket_test_optimized_AR import (
        ARDroidRoboarenaPolicy, WebsocketPolicyServer,
    )
    from eval_utils.policy_server import (
        WebsocketPolicyServer as RoboarenaServer,
        PolicyServerConfig,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    if rank == 0:
        parent_dir = os.path.dirname(model_path)
        date_suffix = datetime.datetime.now().strftime("%Y%m%d")
        checkpoint_name = os.path.basename(model_path)
        output_dir = os.path.join(
            parent_dir,
            f"bi_analysis_{date_suffix}_{args.index}",
            checkpoint_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    wrapper_policy = ARDroidRoboarenaPolicy(
        groot_policy=policy,
        signal_group=signal_group,
        output_dir=output_dir,
    )

    server_config = PolicyServerConfig(
        image_resolution=(180, 320),
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,
        action_space="joint_position",
    )

    try:
        if rank == 0:
            logger.info(f"[BI] 启动 server (host: {hostname}, ip: {local_ip}, port: {args.port})")
            roboarena_server = RoboarenaServer(
                policy=wrapper_policy,
                server_config=server_config,
                host="0.0.0.0",
                port=args.port,
            )
            roboarena_server.serve_forever()
        else:
            policy_metadata = {
                "embodiment": "oxe_droid",
                "model_name": "dreamzero",
                "model_path": model_path,
            }
            server = WebsocketPolicyServer(
                policy=policy,
                host="0.0.0.0",
                port=args.port,
                metadata=policy_metadata,
                output_dir=output_dir,
                signal_group=signal_group,
            )
            asyncio.run(server._worker_loop())
    except KeyboardInterrupt:
        logger.info('[BI] 收到中断信号')
    finally:
        if collector is not None:
            logger.info('[BI] 开始保存分析结果...')
            collector.save_and_plot(args.bi_save_dir)
            collector.remove_hooks()


if __name__ == '__main__':
    main()
