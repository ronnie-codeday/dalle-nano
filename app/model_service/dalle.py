import os
import jax
import jax.numpy as jnp

# Model references

# dalle-mega
DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  # can be wandb artifact or 🤗 Hub or local folder or google bucket
DALLE_COMMIT_ID = None
local_dalle_model = "models/image-gen"

# VQGAN model
VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"
local_vqan_model = "models/vqgan"

# check how many devices are available
jax.local_device_count()
# Load models & tokenizer
from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel

# os.environ['WANDB_MODE'] = 'offline'
print('load model')
if os.path.exists(local_dalle_model):
    print('local model found')
    model, params = DalleBart.from_pretrained(
        local_dalle_model, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    print('local model loaded')
else:
    model, params = DalleBart.from_pretrained(
        DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
    )
    print('save model')
    model.save_pretrained('models/image-gen', params)
    print('models saved to', local_dalle_model)

# Load VQGAN
print('load vqgan')
if os.path.exists(local_vqan_model):
    print('local model found')
    vqgan, vqgan_params = VQModel.from_pretrained(
        local_vqan_model, revision=VQGAN_COMMIT_ID, _do_init=False
    )
    print('local model loaded')
else:
    vqgan, vqgan_params = VQModel.from_pretrained(
        VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
    )
    print('save model')
    vqgan.save_pretrained('models/vqgan', vqgan_params)
    print('saved model to ', local_vqan_model)

from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)

from functools import partial


# model inference

@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
        tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    print('p_generate')
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )


# decode image
@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    print('p_decode')
    return vqgan.decode_code(indices, params=params)


import random

# create a random key

from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

import base64
from io import BytesIO


def generate_predictions(prompt, local_run=False):
    seed = random.randint(0, 2 ** 32 - 1)
    key = jax.random.PRNGKey(seed)
    tokenized_prompts = processor([prompt])

    tokenized_prompt = replicate(tokenized_prompts)
    n_predictions = 8

    # We can customize generation parameters (see https://huggingface.co/blog/how-to-generate)
    gen_top_k = None
    gen_top_p = None
    temperature = None
    cond_scale = 10.0

    images = []
    for _ in trange(max(n_predictions // jax.device_count(), 1)):
        # get a new key
        key, subkey = jax.random.split(key)
        # generate images
        encoded_images = p_generate(
            tokenized_prompt,
            shard_prng_key(subkey),
            params,
            gen_top_k,
            gen_top_p,
            temperature,
            cond_scale,
        )
        # remove BOS
        encoded_images = encoded_images.sequences[..., 1:]
        print(f"encoded images type {type(encoded_images)}")
        # decode images
        decoded_images = p_decode(encoded_images, vqgan_params)
        print(f"decoded images type {type(decoded_images)}")
        decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
        c = 0
        for decoded_img in decoded_images:
            img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
            images.append(img)
            if local_run:
                from datetime import datetime
                now = datetime.now().strftime("%m%dx%YT%H%M%S")
                img.save(f"{now}.bmp")
            c += 1
            print()
    return {"images": images}


def local_run():
    prompt = 'home simpson obama'
    generate_predictions(prompt=prompt, local_run=True)


if __name__ == "__main__":
    local_run()