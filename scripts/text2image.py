from diffusers import StableDiffusionPipeline
import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
unet_model_path = '/data1/JM/code/IP-Adapter-main/exp/base_clip_unet/checkpoint-35000/processed_weight/unet_test'

unet = UNet2DConditionModel.from_pretrained(unet_model_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    "pretrain_models/stable-diffusion-v1-5",
    unet=unet,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

image_list = sorted(os.listdir('/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/text_from_blip2'))

for txt in image_list:
    txt_path = os.path.join('/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/text_from_blip2', txt)
    with open(txt_path, 'r') as f:
        prompt = f.readlines()[0]

    image = pipe(prompt).images[0]  
    image.save(f"/data1/JM/code/IP-Adapter-main/result/base_clip_unet/text2image/{txt.split('.')[0]}.png")
