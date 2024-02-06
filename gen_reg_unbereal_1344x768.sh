#!/bin/bash
. venv/bin/activate

python sdxl_gen_img.py \
	--outdir /home/gazai/opt/DATA/ft_inputs/reg_gen_unbereal_1344x768 \
	--n_iter 3 \
	--W 1344 \
	--H 768 \
	--ckpt /home/gazai/MyPrograms/a1111/stable-diffusion-webui/models/Stable-diffusion/bluePencilXL_v200.safetensors \
	--fp16 \
	--clip_skip 2 \
	--prompt "3way-view,three sided view,full body,simple background,multiple views" \
	--guide_image_path /home/gazai/Downloads/3way_openpose_1344x768.png \
	--control_net_lllite_models /home/gazai/MyPrograms/a1111/stable-diffusion-webui/extensions/sd-webui-controlnet/models/kohya_controllllite_xl_openpose_anime.safetensors
