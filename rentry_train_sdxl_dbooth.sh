#!/usr/bin/env bash
#Set these paths as required:
WEBUI_DIR="/home/gazai/MyPrograms/a1111/stable-diffusion-webui/" # SD-webui folder
MODEL_DIR="/home/gazai/MyPrograms/a1111/stable-diffusion-webui/models/Stable-diffusion/" # Model folder
TRAIN_DIR="/home/gazai/workspace/sd-scripts"  # Kohya scripts folder

# Check if argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <char_name> <optional:base_model>, e.g. char_name=OSIAQUA"
    exit 1
fi

function training_Kohya_lora {
    cd "$TRAIN_DIR" || {
        echo "Folder not found or not accessible."; exit 1
    }

    local base_model="${2:-bluePencilXL_v200}"  # optional arg w/ default value

    local training_set="$1" #Set this to the base folder for a character.
    local ckpt="$MODEL_DIR/${base_model}.safetensors" # Base model(checkpoint) to finetune
    local output="/home/gazai/opt/DATA/model_output/LORA" #Folder to save outputs. WARNING: Will overwrite existing files.
    local dataset_config="/home/gazai/opt/DATA/dataset_configs/$1.toml"

    local learning_rate="0.0001" #Learning rate. Remember this is supposed to be a magnitude larger than a dreambooth equivalent. Worked well for me at this rate.
    local text_encoder_lr="0.00005" #Learning rate for TEXT ENCODER. This is the value suggested in the ninja scrolls. Seems to work better for details.
    local train_batch_size="2" #Amount of images to process at once. I have 8GB of VRAM so I left it at 1, it just worked. Raise if you got more VRAM.
    local num_epochs="6" #Total number of epochs (amount of times the entire set is repeated)
    local save_every_x_epochs="2" #Save checkpoints every X epochs.
    local network_dim="160" #Higher for more resemblance to the training images and bigger file size. 96-192 for characters.
    local scheduler="cosine_with_restarts"

    . venv/bin/activate #Activate your venv before starting.

    accelerate launch --num_cpu_threads_per_process 8 \
	    	      sdxl_train_network.py  \
                      --dataset_config="$dataset_config" \
		      --network_module="networks.lora" \
		      --pretrained_model_name_or_path="$ckpt" \
		      --output_dir="$output" \
		      --output_name="${training_set}_last_e${num_epochs}_n${network_dim}" \
		      --prior_loss_weight=1 \
		      --network_alpha="$network_dim"  \
		      --resolution=1024 \
		      --enable_bucket \
		      --min_bucket_reso=768 \
		      --max_bucket_reso=1024 \
		      --train_batch_size="$train_batch_size"  \
		      --gradient_accumulation_steps=1 \
		      --learning_rate="$learning_rate" \
		      --unet_lr="$learning_rate" \
		      --text_encoder_lr="$text_encoder_lr" \
		      --max_train_epochs="$num_epochs" \
		      --mixed_precision="bf16" \
		      --save_precision="fp16" \
		      --use_8bit_adam \
                      --gradient_checkpointing \
		      --xformers  \
		      --save_every_n_epochs="$save_every_x_epochs" \
		      --save_model_as=safetensors \
		      --clip_skip=2 \
		      --seed=420  \
		      --face_crop_aug_range="2.0,4.0"  \
		      --network_dim="$network_dim" \
		      --max_token_length=150  \
		      --lr_scheduler="$scheduler" \
		      --training_comment="LORA:$training_set"
}

training_Kohya_lora "$@"
