"""
预过滤 chunk 数不足 max_chunk_size 的 step_index。

对每个 (episode, step_index)，复用 _uniform_sample_from_language_ranges 的完整逻辑
计算视频采样的 chunk 数。不足 max_chunk_size 的 step_index 写入 step_filter2.jsonl。

结果会与现有 step_filter.jsonl（如果存在）合并。原文件不会被修改。

用法:
python scripts/data/filter_short_chunks.py
"""

import argparse
import json
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def compute_num_chunks(step_index: int, language_annotations: np.ndarray,
                       trajectory_length: int, max_chunk_size: int) -> int:
    """复用 _uniform_sample_from_language_ranges 的完整逻辑，返回视频 chunk 数。

    这个函数决定了 state/action 的 chunk 数（通过 _current_num_chunks 对齐），
    所以必须和原函数逻辑完全一致。
    """
    first_idx = max(0, min(step_index, trajectory_length - 1))
    target_language = language_annotations[first_idx]
    max_frames = 8 * max_chunk_size + 1
    per_step_offsets = [0, 3, 6, 9, 12, 15, 18, 21]
    sampled_list = []

    def add_step_set(anchor_index):
        if anchor_index < 0 or anchor_index + 23 >= trajectory_length:
            return
        if len(sampled_list) + len(per_step_offsets) > max_frames:
            return
        for offset in per_step_offsets:
            sampled_list.append(anchor_index + offset)

    # 中心 anchor
    add_step_set(first_idx)

    # ±24 步双向扩展
    step = 1
    back_done = False
    fwd_done = False
    while len(sampled_list) < max_frames and (not back_done or not fwd_done):
        if not back_done:
            back_anchor = first_idx - 24 * step
            if back_anchor < 0 or language_annotations[back_anchor] != target_language:
                back_done = True
            else:
                add_step_set(back_anchor)
        if len(sampled_list) >= max_frames:
            break
        if not fwd_done:
            fwd_anchor = first_idx + 24 * step
            if fwd_anchor >= trajectory_length or language_annotations[fwd_anchor] != target_language:
                fwd_done = True
            else:
                add_step_set(fwd_anchor)
        step += 1

    # 去重排序
    if len(sampled_list) == 0:
        return 0
    unique_sorted = np.array(sorted(set(sampled_list)), dtype=int)
    if unique_sorted.size > max_frames:
        unique_sorted = unique_sorted[:max_frames]

    # 凑 8n+1 格式
    if unique_sorted.size > 0:
        last_idx = unique_sorted[-1]
        additional_idx = last_idx + 3
        if additional_idx < trajectory_length and unique_sorted.size < max_frames:
            unique_sorted = np.append(unique_sorted, additional_idx)
        else:
            if unique_sorted.size <= 8:
                return 0
            unique_sorted = unique_sorted[:-7]

    assert unique_sorted.size % 8 == 1, f"unique_sorted size {unique_sorted.size} is not 8n+1"

    # 原函数在此处记录 chunk 数供 state/action 对齐：
    # self._current_num_chunks[first_idx] = num_video_chunks
    # get_state/get_action 会读取该值，将采样 chunk 数限制为与视频一致。
    # 因此视频 chunk 数决定了整个样本的 shape（包括 state 和 action）。
    # 过滤脚本只需要返回 chunk 数，不需要实际记录。
    num_video_chunks = (unique_sorted.size - 1) // 8
    return num_video_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/datadrive/wjy/dataset/DreamZero-DROID-Data")
    parser.add_argument("--max_chunk_size", type=int, default=4)
    parser.add_argument("--language_col", type=str,
                        default="annotation.language.language_instruction")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    meta_dir = dataset_path / "meta"
    max_chunk_size = args.max_chunk_size

    # 加载现有 step_filter（如果有）
    old_filter_path = meta_dir / "step_filter.jsonl"
    existing_filter = {}
    if old_filter_path.exists():
        with open(old_filter_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                existing_filter[entry["episode_index"]] = set(entry["step_indices"])
        print(f"Loaded existing step_filter: {len(existing_filter)} episodes")

    # 扫描所有 parquet 文件
    parquet_files = sorted(glob.glob(
        str(dataset_path / "data" / "chunk-*" / "*.parquet")
    ))
    print(f"Found {len(parquet_files)} episode parquet files")

    total_steps = 0
    total_filtered_new = 0
    total_filtered_existing = 0
    filter_result = {}

    for pf in tqdm(parquet_files, desc="Scanning episodes"):
        table = pq.read_table(pf, columns=[args.language_col])
        lang = np.array(table[args.language_col].to_pylist())
        traj_len = len(lang)

        fname = os.path.basename(pf)
        episode_index = int(fname.replace("episode_", "").replace(".parquet", ""))

        existing_set = existing_filter.get(episode_index, set())
        new_filtered = []

        for step_idx in range(traj_len):
            total_steps += 1
            if step_idx in existing_set:
                total_filtered_existing += 1
                continue
            n_chunks = compute_num_chunks(step_idx, lang, traj_len, max_chunk_size)
            if n_chunks < max_chunk_size:
                new_filtered.append(step_idx)
                total_filtered_new += 1

        merged = sorted(existing_set | set(new_filtered))
        if merged:
            filter_result[episode_index] = merged

    total_filtered = total_filtered_existing + total_filtered_new
    total_remaining = total_steps - total_filtered
    print(f"\n===== Summary =====")
    print(f"Total steps:              {total_steps}")
    print(f"Already filtered:         {total_filtered_existing}")
    print(f"Newly filtered (chunks<{max_chunk_size}): {total_filtered_new}")
    print(f"Total filtered:           {total_filtered}")
    print(f"Remaining valid steps:    {total_remaining}")
    print(f"Filter rate:              {total_filtered / total_steps * 100:.2f}%")

    out_path = meta_dir / "step_filter2.jsonl"
    if old_filter_path.exists():
        shutil.copy2(old_filter_path, out_path)
        print(f"\nCopied {old_filter_path} -> {out_path}")

    with open(out_path, "w") as f:
        for ep_idx in sorted(filter_result.keys()):
            entry = {
                "episode_index": ep_idx,
                "step_indices": filter_result[ep_idx],
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Written to {out_path}")
    print(f"Episodes with filters: {len(filter_result)}")


if __name__ == "__main__":
    main()
