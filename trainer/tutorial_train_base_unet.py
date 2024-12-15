import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())
from ip_adapter.utils import is_torch2_available
from diffusers.utils.import_utils import is_xformers_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

os.environ['NCCL_P2P_DISABLE'] = "1"
os.environ['NCCL_IB_DISABLE'] = "1"

# Dataset
class MyDataset(torch.utils.data.Dataset):

    def __init__(self, json_file, tokenizer, size=512, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05):
        super().__init__()

        self.tokenizer = tokenizer
        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        # list of dict: [{"source":"路径", "target":"路径", "mask":"路径", "object":"路径", text": "A dog"}]
        self.data = json.load(open(json_file)) # list of dict: [{"image_file": "1.png", "text": "A dog"}]

        self.transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.clip_image_processor = CLIPImageProcessor()
        
    def __getitem__(self, idx):
        item = self.data[idx] 
        source_image_path = item['source']
        target_image_path = item['target']
        mask_image_path = item['mask']
        object_image_path = item['object']
        text = item["text"]
        
        # read image
        source_image = Image.open(source_image_path)
        source_image = self.transform(source_image.convert("RGB"))

        target_image = Image.open(target_image_path)
        target_image = self.transform(target_image.convert("RGB"))

        object_image = self.random_flip(Image.open(object_image_path))
        object_image = self.transform(object_image.convert("RGB"))    

        mask_image = Image.open(mask_image_path)
        mask_image = self.transform(mask_image.convert("L"))    

        # get text and tokenize
        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            "source_image": source_image,
            "target_image": target_image,
            "mask_image": mask_image,
            "object_image": object_image,
            "text_input_ids": text_input_ids,
        }

    def __len__(self):
        return len(self.data)

    def random_flip(self, image):
        if random.choice([True, False]):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        if random.choice([True, False]):
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        return image
    

def collate_fn(data):
    source_images = torch.stack([example["source_image"] for example in data])
    target_images = torch.stack([example["target_image"] for example in data])
    mask_images = torch.stack([example["mask_image"] for example in data])
    object_images = torch.stack([example["object_image"] for example in data])
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)

    return {
            "source_image": source_images,
            "target_image": target_images,
            "mask_image": mask_images,
            "object_image": object_images,
            "text_input_ids": text_input_ids,
           }
    

class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        return noise_pred

    
def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/data1/JM/code/IP-Adapter-main/pretrain_models/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default="/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/data.json",
        help="Training data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/test",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=args.output_dir)  # 将日志文件保存到 output_dir

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to('cuda')
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ip_adapter = IPAdapter(unet)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    #unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # optimizer
    params_to_opt = itertools.chain(
        unet.parameters()
    )

    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # dataloader
    train_dataset = MyDataset(args.data_json_file, tokenizer=tokenizer, size=args.resolution)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    global_step = 0

    # 外层是 epoch 进度条
    for epoch in range(0, args.num_train_epochs):

        # 创建一个 tqdm 进度条，用于显示每个 epoch 内的训练进度
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_train_epochs}", unit="batch") as tbar:
            for step, batch in enumerate(tbar):
                # 使用 accelerator.accumulate 优化器加速
                with accelerator.accumulate(ip_adapter):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["target_image"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        source_latents = vae.encode(batch["source_image"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()
                        object_latents = vae.encode(batch["object_image"].to(accelerator.device, dtype=weight_dtype)).latent_dist.sample()

                        latents = latents * vae.config.scaling_factor
                        source_latents = source_latents * vae.config.scaling_factor
                        object_latents = object_latents * vae.config.scaling_factor

                    # import torchvision
                    # torchvision.utils.save_image(batch["target_image"], 'target_image.png')
                    # torchvision.utils.save_image(batch["source_image"], 'source_image.png')
                    # torchvision.utils.save_image(batch["object_image"], 'object_image.png')
                    # torchvision.utils.save_image(batch["mask_image"], 'mask_image.png')
                    # exit()

                    masks = torch.nn.functional.interpolate(
                        batch["mask_image"], 
                        size=(
                            latents.shape[-2], 
                            latents.shape[-1]
                        )
                    )
                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["text_input_ids"].to(accelerator.device))[0]

                    noisy_latents_concat = torch.cat([noisy_latents, masks, source_latents, object_latents], dim=1)
                    noise_pred = ip_adapter(noisy_latents_concat, timesteps, encoder_hidden_states)

                    # Calculate loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    # Gather the losses across all processes for logging (if we use distributed training)
                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()

                    # Update progress bar description with the loss
                    tbar.set_postfix(loss=avg_loss)

                    # Backpropagate
                    accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()

                    # 每 100 步记录一次损失
                    if global_step % 300 == 0:
                        writer.add_scalar('Loss/train', avg_loss, global_step)

                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path, safe_serialization=False)

                begin = time.perf_counter()

    # 关闭 TensorBoard writer
    writer.close()

                
if __name__ == "__main__":
    main()    
