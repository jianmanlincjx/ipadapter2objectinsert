import os
import torch
from safetensors.torch import save_file

# 定义路径
ckpt = "/data1/JM/code/IP-Adapter-main/exp/base_unet/checkpoint-85000/pytorch_model.bin"  # 使用 safetensors 格式
save_unet_dir = os.path.join(os.path.dirname(ckpt), "processed_weight", 'unet')
os.makedirs(save_unet_dir, exist_ok=True)

sd = torch.load(ckpt)  # 使用 safetensors 加载模型

# 创建三个字典，分别存储不同的权重
image_proj_sd = {}
ip_sd = {}
unet_sd = {}

for k in sd:
    if k.startswith("unet"):
        unet_sd[k.replace("unet.", "")] = sd[k]

combined_ip_adapter = {**image_proj_sd, **ip_sd}

torch.save(unet_sd, f"{save_unet_dir}/diffusion_pytorch_model.bin")

print("权重已分离并保存")
