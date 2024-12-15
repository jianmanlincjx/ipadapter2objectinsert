import torch

# 加载两个权重文件
weights_1 = torch.load('/data1/JM/code/IP-Adapter-main/exp/base_unet/checkpoint-10/pytorch_model.bin')
weights_2 = torch.load('/data1/JM/code/IP-Adapter-main/exp/base_unet/checkpoint-20/pytorch_model.bin')

# 获取模型权重的名字列表
keys_1 = set(weights_1.keys())
keys_2 = set(weights_2.keys())

# 计算键的差异
common_keys = keys_1.intersection(keys_2)
diff_keys = keys_1.symmetric_difference(keys_2)

# 输出差异的键
if diff_keys:
    print("不同的参数名：", diff_keys)
else:
    print("两个检查点的参数名完全一致。")

# 对比每个共同的参数值
for key in common_keys:
    weight_1 = weights_1[key]
    weight_2 = weights_2[key]

    if weight_1.shape == weight_2.shape:
        # 比较两个权重是否相同（考虑浮动误差）
        diff = torch.abs(weight_1 - weight_2).max().item()
        if diff > 1e-5:
            print(f"参数 {key} 在两个检查点中的最大差异是: {diff}")
        else:
            print(f"参数 {key} 在两个检查点中的值几乎相同。")
    else:
        print(f"参数 {key} 的形状不同，无法比较。")
