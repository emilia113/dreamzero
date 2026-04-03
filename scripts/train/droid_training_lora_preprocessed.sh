#!/bin/bash
# DreamZero DROID Training Script (Preprocessed Text Embeddings)
#
# Usage:
#   # Set your dataset path and output directory, then run:
#   bash scripts/train/droid_training_lora_preprocessed.sh
#
# Prerequisites:
#   - DROID dataset in LeRobot format at DROID_DATA_ROOT (with preprocessed text embeddings)
#     Download: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir ./data/droid_lerobot
#     Or convert from scratch: see scripts/data/convert_droid.py
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P

export HYDRA_FULL_ERROR=1

# ============ USER CONFIGURATION ============
# Dataset path (DROID in LeRobot format)
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"./data/droid_lerobot"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_droid_lora"}

# Number of GPUs to use
NUM_GPUS=${NUM_GPUS:-8}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    echo "Download with: huggingface-cli download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir $DROID_DATA_ROOT"
    exit 1
fi

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative_preprocessed \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_preprocessed \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=100 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=no \
    droid_data_root=$DROID_DATA_ROOT \
    text_embeddings_dir=$DROID_DATA_ROOT/text_embeddings \
    dit_version=$WAN_CKPT_DIR \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth