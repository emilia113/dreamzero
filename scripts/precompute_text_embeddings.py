"""
预计算 DROID 数据集的 text embeddings，每个 task_index 存一个 .pt 文件。
训练时按 task_index 加载对应文件，跳过 UMT5-XXL text encoder，节省 GPU 显存。

使用方法:
    python scripts/precompute_text_embeddings.py \
        --data_root ./dataset/DreamZero-DROID-Data \
        --tokenizer_path ./ckpt/umt5-xxl \
        --text_encoder_pretrained_path ./ckpt/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth \
        --output_dir ./dataset/droid_text_embeddings/ \
        --max_length 512

输出目录结构:
    droid_text_embeddings/
        000000.pt   # task_index=0 的 embedding, shape [512, 4096] bfloat16
        000001.pt   # task_index=1
        ...
        metadata.json  # 元信息

训练时使用:
    emb = torch.load(f"droid_text_embeddings/{task_index:06d}.pt")  # [512, 4096]
"""

import argparse
import os
import json
import torch
from tqdm import tqdm


def load_all_tasks(data_root: str) -> list:
    """
    从 DROID 数据集的 meta/tasks.jsonl 按 task_index 顺序加载所有 prompt。

    数据格式: LeRobot v2.0
      - meta/tasks.jsonl 每行: {"task_index": N, "task": "prompt string"}
      - 共 123259 条，每条都是唯一的
      - parquet 中 annotation.language.language_instruction 存的是 task_index (int)

    来源: meta/info.json 的 features 字段
    """
    tasks_path = os.path.join(data_root, "meta", "tasks.jsonl")
    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"tasks.jsonl not found at {tasks_path}")

    tasks = {}
    with open(tasks_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_index = entry["task_index"]
            task = entry.get("task", "")
            tasks[task_index] = task

    max_index = max(tasks.keys())
    task_list = [tasks.get(i, "") for i in range(max_index + 1)]
    return task_list


def apply_droid_template(raw_prompt: str) -> str:
    """
    对 DROID embodiment 的 prompt 添加模板前缀。

    来源: dreamzero_cotrain.py collate(), line 112-118
    当 embodiment_id == oxe_droid 时，先 .lower() 再拼接此模板。
    """
    processed = raw_prompt.lower()
    full_prompt = (
        "A multi-view video shows that a robot "
        + processed
        + " The video is split into three views: "
        "The top view shows the camera view from the robot's wrist, "
        "the bottom-left view shows the camera view from the left exterior camera, "
        "and the bottom-right view shows the camera view from the right exterior camera. "
        "During training, one of the two bottom exterior views may be a black screen (dropped view). "
        "The robot "
        + processed
    )
    return full_prompt


def encode_prompt(text_encoder, input_ids, attention_mask):
    """
    用 text encoder 编码，和训练时逻辑完全一致。

    来源: wan_flow_matching_action_tf.py encode_prompt(), line 547-553
    """
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    with torch.no_grad():
        prompt_emb = text_encoder(input_ids, attention_mask)
    prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
    for i, v in enumerate(seq_lens):
        prompt_emb[i, v:] = 0
    return prompt_emb


def main():
    parser = argparse.ArgumentParser(description="Precompute text embeddings for DROID dataset")
    parser.add_argument("--data_root", type=str, default="./dataset/DreamZero-DROID-Data",
                        help="DROID 数据集根目录")
    parser.add_argument("--tokenizer_path", type=str, default="./ckpt/umt5-xxl",
                        help="Tokenizer 路径 (来源: dreamzero_cotrain.yaml tokenizer_path, "
                             "训练时命令行传入 ./ckpt/umt5-xxl)")
    parser.add_argument("--text_encoder_pretrained_path", type=str,
                        default="./ckpt/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Text encoder 权重路径 (来源: wan_flow_matching_action_tf.yaml "
                             "text_encoder_pretrained_path, 训练时命令行传入)")
    parser.add_argument("--output_dir", type=str, default="./dataset/droid_text_embeddings",
                        help="输出目录，每个 task_index 存一个 .pt 文件")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Tokenizer max_length (来源: dreamzero_cotrain.yaml max_length)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="编码时的 batch size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="编码设备")
    args = parser.parse_args()

    # ========== Step 1: 从 meta/tasks.jsonl 加载所有 prompt ==========
    print("Step 1: Loading all tasks from meta/tasks.jsonl...")
    task_list = load_all_tasks(args.data_root)
    num_tasks = len(task_list)
    print(f"  Loaded {num_tasks} tasks (task_index 0 ~ {num_tasks - 1})")

    for i in range(min(3, num_tasks)):
        print(f"  [task_index={i}] {task_list[i][:80]}")

    # ========== Step 2: 应用 DROID 模板 ==========
    # 来源: dreamzero_cotrain.py collate(), line 112-118
    print("\nStep 2: Applying DROID template...")
    templated_list = [apply_droid_template(p) for p in task_list]
    print(f"  Example: {templated_list[1][:100]}...")

    # ========== Step 3: 初始化 tokenizer ==========
    # 使用 dreamzero 源代码的 HuggingfaceTokenizer，和训练时完全一致
    # 来源: dreamzero_cotrain.py HuggingfaceTokenizer, line 38-89
    #   DefaultDataCollator.__init__ (line 218-220):
    #     self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=max_length, clean='whitespace')
    print(f"\nStep 3: Initializing tokenizer (same as DefaultDataCollator)...")
    from groot.vla.model.dreamzero.transform.dreamzero_cotrain import HuggingfaceTokenizer
    tokenizer = HuggingfaceTokenizer(
        name=args.tokenizer_path,
        seq_len=args.max_length,
        clean='whitespace',
    )

    # ========== Step 4: 初始化 text encoder ==========
    # 使用 dreamzero 源代码的 instantiate + load_state_dict，和训练时完全一致
    # 来源: wan_flow_matching_action_tf.py __init__():
    #   line 176: self.text_encoder = instantiate(config.text_encoder_cfg)
    #   line 244-248: text_enc_path = ensure_file(...); self.text_encoder.load_state_dict(...)
    print(f"\nStep 4: Initializing text encoder (same as WANPolicyHead.__init__)...")
    from hydra.utils import instantiate
    # 构造和 yaml 中 text_encoder_cfg 一样的配置字典
    # 来源: wan_flow_matching_action_tf.yaml line 77-80
    text_encoder_cfg = {
        "_target_": "groot.vla.model.dreamzero.modules.wan_video_text_encoder.WanTextEncoder",
        "_convert_": "object",
        "text_encoder_pretrained_path": args.text_encoder_pretrained_path,
    }
    text_encoder = instantiate(text_encoder_cfg)
    # 加载权重，和 wan_flow_matching_action_tf.py line 244-248 一致
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import ensure_file
    text_enc_path = ensure_file(
        text_encoder.text_encoder_pretrained_path,
        "models_t5_umt5-xxl-enc-bf16.pth",
    )
    text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))
    text_encoder = text_encoder.to(device=args.device, dtype=torch.bfloat16)
    text_encoder.eval()
    print(f"  Text encoder loaded from {text_enc_path}")

    # ========== Step 5: 分 batch 编码并逐条保存 ==========
    # tokenize 来源: dreamzero_cotrain.py collate() line 153
    #   ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
    # encode 来源: wan_flow_matching_action_tf.py encode_prompt() line 547-553
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nStep 5: Encoding and saving to {args.output_dir}/")
    print(f"  {num_tasks} tasks, batch_size={args.batch_size}")

    for start in tqdm(range(0, num_tasks, args.batch_size), desc="Encoding"):
        end = min(start + args.batch_size, num_tasks)
        batch_texts = templated_list[start:end]

        # Tokenize — 调用方式和 collate() line 153 完全一致
        input_ids, attention_mask = tokenizer(
            batch_texts, return_mask=True, add_special_tokens=True,
        )
        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)

        # Encode — 和 wan_flow_matching_action_tf.py encode_prompt() line 547-553 完全一致
        batch_emb = encode_prompt(text_encoder, input_ids, attention_mask)  # [B, 512, 4096]

        # 逐条保存
        for i in range(batch_emb.shape[0]):
            task_index = start + i
            save_path = os.path.join(args.output_dir, f"{task_index:06d}.pt")
            torch.save(batch_emb[i].cpu(), save_path)  # [512, 4096] bfloat16

    # ========== Step 6: 保存元信息 ==========
    del text_encoder
    torch.cuda.empty_cache()

    metadata = {
        "num_tasks": num_tasks,
        "max_length": args.max_length,
        "embedding_shape": [args.max_length, 4096],
        "dtype": "bfloat16",
        "tokenizer_path": args.tokenizer_path,
        "text_encoder_pretrained_path": args.text_encoder_pretrained_path,
        "template": "droid",
        "file_format": "{task_index:06d}.pt",
    }
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    total_size_gb = num_tasks * args.max_length * 4096 * 2 / (1024**3)
    print(f"\nDone! {num_tasks} embeddings saved to {args.output_dir}/")
    print(f"  Each file: [{args.max_length}, 4096] bfloat16 (~4MB)")
    print(f"  Total disk: ~{total_size_gb:.0f} GB")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
