import torch

# 权重文件路径
checkpoint1_path = '/data1/JM/code/IP-Adapter-main/exp/base_clip_unet/checkpoint-10/processed_weight/unet/diffusion_pytorch_model.bin'
checkpoint2_path = '/data1/JM/code/IP-Adapter-main/exp/base_clip_unet/checkpoint-20/processed_weight/unet/diffusion_pytorch_model.bin'

# 加载权重文件
checkpoint1 = torch.load(checkpoint1_path)
checkpoint2 = torch.load(checkpoint2_path)

# 确保两个权重文件包含相同的键
keys1 = set(checkpoint1.keys())
keys2 = set(checkpoint2.keys())

if keys1 != keys2:
    print(f"警告：两个权重文件的键不同！\n"
          f"在文件 1 中有 {keys1 - keys2}，在文件 2 中有 {keys2 - keys1}")
else:
    print("两个权重文件的键完全一致。")

# 比较相同键对应的值
for key in keys1:
    if torch.equal(checkpoint1[key], checkpoint2[key]):
        print(f"权重 {key} 的值一致。")
    else:
        print(f"权重 {key} 的值不一致。")
