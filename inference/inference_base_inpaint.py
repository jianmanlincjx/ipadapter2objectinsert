from curses import noecho
import os
import sys
sys.path.append(os.getcwd())

import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from PIL import Image
from ip_adapter import IPAdapter
from torchvision import transforms
import cv2

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator

base_model_path = "pretrain_models/stable-diffusion-v1-5"
vae_model_path = "pretrain_models/models--stabilityai--sd-vae-ft-mse"
image_encoder_path = "pretrain_models/image_encoder"
device = "cuda"

unet_model_path = "/data1/JM/code/IP-Adapter-main/exp/base_unet/checkpoint-85000/processed_weight/unet"

base_dir = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K_inpaint'
image_dir = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K_inpaint/source_processed'
save_dir = '/data1/JM/code/IP-Adapter-main/result/base_unet/train'
os.makedirs(save_dir, exist_ok=True)

transforms_ = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path, torch_dtype=torch.float16).to(dtype=torch.float16).cuda()
unet = UNet2DConditionModel.from_pretrained(unet_model_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None).to(dtype=torch.float16).cuda()

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    feature_extractor=None,
    safety_checker=None
).to('cuda')


image_name_list = [i.split('.')[0] for i in os.listdir(image_dir)]
for name in image_name_list[:100]:
    object_image_path = f'{base_dir}/object_processed/{name}.png'
    source_image_path = f'{base_dir}/source_processed/{name}.png'
    target_image_path = f'{base_dir}/target_processed/{name}.png'
    mask_image_path = f'{base_dir}/mask_processed/{name}.png'
    prompt_txt_path = f'{base_dir}/text/{name}.txt'

    with open(prompt_txt_path, 'r') as f:
        prompt_ = f.readlines()
        prompt = prompt_[0] if len(prompt_) > 0 else None

    object_image = Image.open(object_image_path)
    object_image = transforms_(object_image.convert("RGB")).unsqueeze(0)

    source_image = Image.open(source_image_path)
    source_image = transforms_(source_image.convert("RGB")).unsqueeze(0)

    mask_image = Image.open(mask_image_path)
    mask_image = transforms_(mask_image.convert("L")).unsqueeze(0)    

    with torch.no_grad():
        source_latents = vae.encode(source_image.to(dtype=torch.float16).cuda()).latent_dist.sample()
        source_latents = source_latents * vae.config.scaling_factor

        object_latents = vae.encode(object_image.to(dtype=torch.float16).cuda()).latent_dist.sample()
        object_latents = object_latents * vae.config.scaling_factor

    mask_latent = torch.nn.functional.interpolate(
                        mask_image, 
                        size=(
                            source_latents.shape[-2], 
                            source_latents.shape[-1]
                        )
                    )
    
    noise = torch.randn_like(source_latents).to(pipe.dtype)
    noisy_latents = noise_scheduler.add_noise(source_latents, noise, torch.tensor(899))

    mask_latent = torch.cat([mask_latent] * 2).to(dtype=torch.float16).cuda()
    source_latents = torch.cat([source_latents] * 2).to(dtype=torch.float16).cuda()
    object_latents = torch.cat([object_latents] * 2).to(dtype=torch.float16).cuda()

    condition_list = [mask_latent, source_latents, object_latents]

    if prompt is None:
        prompt = "best quality, high quality"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

    with torch.inference_mode():
        prompt_embeds_, negative_prompt_embeds_ = pipe.encode_prompt(
            prompt,
            device='cuda',
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )

    generator = get_generator(7777, 'cuda')

    images = pipe(
        prompt_embeds=prompt_embeds_,
        negative_prompt_embeds=negative_prompt_embeds_,
        guidance_scale=7.5,
        num_inference_steps=50,
        generator=generator, latent_condition=condition_list, latents=noisy_latents
    ).images[0]

    ###################################################################################
    ###################################################################################
    ###################################################################################
    source_image_ = cv2.imread(source_image_path)
    source_image_ = cv2.cvtColor(source_image_, cv2.COLOR_BGR2RGB)
    source_image_ = torch.from_numpy(source_image_).unsqueeze(0).permute(0, 3, 1, 2)
    
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_image = torch.from_numpy(target_image).unsqueeze(0).permute(0, 3, 1, 2)

    if isinstance(source_image, torch.Tensor):
        source_image_ = transforms.ToPILImage()(source_image_.squeeze(0))

    if isinstance(images, torch.Tensor):
        images = transforms.ToPILImage()(images.squeeze(0))

    if isinstance(target_image, torch.Tensor):
        target_image = transforms.ToPILImage()(target_image.squeeze(0))

    concatenated_image = Image.new('RGB', (source_image_.width + images.width + target_image.width, source_image_.height))
    concatenated_image.paste(source_image_, (0, 0))
    concatenated_image.paste(images, (source_image_.width, 0))
    concatenated_image.paste(target_image, (source_image_.width + images.width, 0))
    concatenated_image.save(f"{save_dir}/{name}.png")
