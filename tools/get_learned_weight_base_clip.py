import os
import torch
from safetensors.torch import load_file, save_file

# 定义路径
ckpt = "exp/base_clip/checkpoint-20000/pytorch_model.bin"  # 使用 safetensors 格式
save_unet_dir = os.path.join(os.path.dirname(ckpt), "processed_weight", 'unet')
save_ipadapter_dir = os.path.join(os.path.dirname(ckpt), "processed_weight", 'ipadapter')
os.makedirs(save_unet_dir, exist_ok=True)
os.makedirs(save_ipadapter_dir, exist_ok=True)

sd = torch.load(ckpt)  # 使用 safetensors 加载模型

# 创建三个字典，分别存储不同的权重
image_proj_sd = {}
ip_sd = {}
unet_sd = {}

for k in sd:
    if k.startswith("unet"):
        if k == "unet.conv_in.weight":
            unet_sd[k.replace("unet.", "")] = sd[k]
    elif k.startswith("image_proj_model"):
        image_proj_sd[k.replace("image_proj_model.", "")] = sd[k]
    elif k.startswith("adapter_modules"):
        ip_sd[k.replace("adapter_modules.", "")] = sd[k]

combined_ip_adapter = {**image_proj_sd, **ip_sd}

save_file(unet_sd, f"{save_unet_dir}/diffusion_pytorch_model.bin")

torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, f"{save_ipadapter_dir}/ip_adapter.bin")

print("权重已分离并保存")
