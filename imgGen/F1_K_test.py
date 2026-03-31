import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("/home/coder/passenger/FLUX1-dev/culture/gen_images/11.jpg")

image = pipe(
  image=input_image,
  prompt="The camera module should be wide enough to span the full horizontal width of the phone. The flash and LiDAR sensor should be positioned on the right side.",
  guidance_scale=2.5
).images[0]
image.save("ct-11.png")