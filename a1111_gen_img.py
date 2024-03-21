"""
Ad-hoc script to generate images using a1111
Don't use as-is! Needs improvement
"""
import io
import os
import argparse
import base64
import requests
from tqdm import tqdm

from PIL import Image

#DEFAULT_ENDPOINT = 'http://genai.kccheng.com:7860'
DEFAULT_ENDPOINT = 'http://127.0.0.1:7860'

# TODO: Make these command line args
PROMPT = "a guy"
TOTAL_IMGS = 200
OUTPUT_DIR = #"/home/gazai/opt/DATA/ft_inputs/reg_gen_whomor_ch1_style_guy"

def _parse_response(resp, batch_size=None):
    # Get list of PIL images
    images_raw = resp['images']#[:batch_size]
    images = [Image.open(io.BytesIO(base64.b64decode(image_raw.split(",", 1)[0])))
        for image_raw in images_raw]

    if batch_size:
        return images[:batch_size], resp['info']
    else:
        return images[0], resp['info']


def gen_tex2img(
        prompt,
        neg_prompt="",
        height=1024,
        width=1024,
        batch_size=1,
        n_iter=1,
        cfg_scale=7,
        steps=20,
        #seed=-1,
        url=DEFAULT_ENDPOINT,
        **kwargs):
    """
    Generic txt2img
    """
    # A1111 payload
    #if seed == -1:
    #    seed = get_random_seed()

    payload = {
        "prompt": prompt,
        "negative_prompt": neg_prompt,
        "height": height,
        "width": width,
        "n_iter": n_iter,
        "sampler_name": "DPM++ SDE Karras",
        "batch_size": batch_size,
        "steps": steps,
        #"seed": seed,
        "cfg_scale": cfg_scale,
        # "save_images": True,
    }
    # Trigger Generation
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    # Read results
    resp = response.json()
    parsed_resp = _parse_response(resp, batch_size=batch_size)

    return parsed_resp

if __name__ == "__main__":
    for i in tqdm(range(TOTAL_IMGS)):
        output_images, info = gen_tex2img(PROMPT)
        output_images[0].save(os.path.join(OUTPUT_DIR, f"{i}.png"))
