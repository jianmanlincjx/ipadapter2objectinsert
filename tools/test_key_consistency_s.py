import torch
from safetensors.torch import load_file

# 加载两个 safetensors 权重文件
checkpoint_10 = load_file("/data1/JM/code/IP-Adapter-main/exp/test/checkpoint-10/model.safetensors")
checkpoint_20 = load_file("/data1/JM/code/IP-Adapter-main/exp/test/checkpoint-20/model.safetensors")

# 找出两个文件中所有的键
keys_10 = set(checkpoint_10.keys())
keys_20 = set(checkpoint_20.keys())

# 找出两个文件中相同的键
common_keys = keys_10.intersection(keys_20)

# 检查哪些键的 value 值不一致
inconsistent_keys = []

for key in common_keys:
    # 比较两个模型中该键的 tensor 是否一致
    if not torch.equal(checkpoint_10[key], checkpoint_20[key]):
        inconsistent_keys.append(key)

# 打印不一致的键
if inconsistent_keys:
    print("以下键的值不一致：")
    for key in inconsistent_keys:
        print(key)
else:
    print("所有的键值一致。")
