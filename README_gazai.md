## Generate regularization images
```
python sdxl_gen_img.py --ckpt ~/MyPrograms/a1111/stable-diffusion-webui/models/Stable-diffusion/bluePencilXL_v200.safetensors --outdir /home/gazai/opt/DATA/ft_inputs/reg_gen_military_uniform_sdxl --xformers --fp16 --images_per_prompt 5 --prompt "a military_uniform,a guy,single shot full body,"
```

## Run commands in background
```
nohup $COMMAND > run.log 2>&1 &
```
