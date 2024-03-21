#!/usr/bin/env bash
#Set these paths as required:
WEBUI_DIR="/home/gazai/opt/stable-diffusion-webui/" # SD-webui folder
MODEL_DIR="/home/gazai/opt/stable-diffusion-webui/models/Stable-diffusion/" # Model folder
TRAIN_DIR="/home/gazai/workspace/sd-scripts"  # Kohya scripts folder

# Check if argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <dataset_config_name> <optional:base_model>, e.g. dataset_config_name=foo (referring to foo.toml)"
    exit 1
fi

function training_controlnet_lllite {
    cd "$TRAIN_DIR" || {
        echo "Folder not found or not accessible."; exit 1
    }

    local base_model="${2:-bluePencilXL_v200}"  # optional arg w/ default value

    local training_set="$1" #Set this to the base folder for a character.
    local ckpt="$MODEL_DIR/${base_model}.safetensors" # Base model(checkpoint)
    local output="/home/gazai/opt/DATA/model_output/controlnet_lllite" #Folder to save outputs. WARNING: Will overwrite existing files.
    local dataset_config="/home/gazai/opt/DATA/dataset_configs/$1.toml"

    local learning_rate="0.0001"
    local text_encoder_lr="0.00005"
    local train_batch_size="2"
    local num_epochs="6"
    local save_every_x_epochs="99999" #Save checkpoints every X epochs.
    local network_dim="64"
    local cond_emb_dim="32"
    local scheduler="cosine_with_restarts"

    . venv/bin/activate #Activate your venv before starting.

    accelerate launch --num_cpu_threads_per_process 8 \
              sdxl_train_control_net_lllite.py  \
              --dataset_config="$dataset_config" \
              --caption_extension=".txt" \
              --pretrained_model_name_or_path="$ckpt" \
              --output_dir="$output" \
              --output_name="${training_set}_e${num_epochs}_n${network_dim}" \
              --resolution=1024 \
              --enable_bucket \
              --min_bucket_reso=640 \
              --max_bucket_reso=1536 \
              --train_batch_size="$train_batch_size"  \
              --gradient_accumulation_steps=1 \
              --learning_rate="$learning_rate" \
              --max_train_epochs="$num_epochs" \
              --mixed_precision="bf16" \
              --save_precision="fp16" \
              --use_8bit_adam \
              --gradient_checkpointing \
              --xformers  \
              --save_every_n_epochs="$save_every_x_epochs" \
              --save_model_as=safetensors \
              --seed=420  \
              --network_dim="$network_dim" \
              --cond_emb_dim="$cond_emb_dim" \
              --max_token_length=150  \
              --lr_scheduler="$scheduler"
}

training_controlnet_lllite "$@"
