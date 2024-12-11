import torch

# 加载两个权重文件
weights_1 = torch.load('/data1/JM/code/IP-Adapter-main/weight_modify2inchannel_4.pth')
weights_2 = torch.load('/data1/JM/code/IP-Adapter-main/weight_modify2inchannel_12.pth')

# 提取权重文件的键
keys_1 = set(weights_1.keys())  # 第一个权重文件的所有键
keys_2 = set(weights_2.keys())  # 第二个权重文件的所有键

# 找到共同的键（即两个字典中都有的键）
common_keys = keys_1.intersection(keys_2)

# 分类：一致的键和值是否相同；不一致的键
consistent_keys = []
inconsistent_keys = []

# 比较这些共同的键的值
for key in common_keys:
    # 检查两个tensor的形状是否相同
    if weights_1[key].shape == weights_2[key].shape:
        # 如果形状相同，则比较值是否一致
        if torch.all(weights_1[key] == weights_2[key]):  # 检查两个值是否相同
            consistent_keys.append(key)
        else:
            inconsistent_keys.append(key)
    else:
        # 如果形状不一致，则标记为不一致
        inconsistent_keys.append(key)

# 打印结果
print("一致的key和值相同的：")
for key in consistent_keys:
    print(key)

print("\n不一致的key：")
for key in inconsistent_keys:
    print(key)
