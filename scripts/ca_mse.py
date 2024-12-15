import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# 文件夹路径
folder_result = '/data1/JM/code/IP-Adapter-main/result/base_clip/result'
folder_label = '/data1/JM/code/IP-Adapter-main/result/base_clip/label'
folder_mask = '/data1/JM/code/IP-Adapter-main/result/base_clip/mask'

# 获取文件夹中的所有文件
files_result = sorted(os.listdir(folder_result))
files_label = sorted(os.listdir(folder_label))
files_mask = sorted(os.listdir(folder_mask))

# 确保两个文件夹的文件数量一致
if len(files_result) != len(files_label):
    print("result和label文件夹中的文件数量不一致！")
else:
    mse_total = 0  # 累加MSE Loss
    count = 0  # 计数器
    for file_result, file_label in zip(files_result, files_label):
        # 构造mask文件名（假设mask文件与result和label文件命名相同）
        mask_filename = file_result  # 假设mask文件与result文件同名
        if mask_filename not in files_mask:
            print(f"未找到对应的mask文件：{mask_filename}")
            continue

        # 读取result、label和mask图像
        img_result = np.array(Image.open(os.path.join(folder_result, file_result)))
        img_label = np.array(Image.open(os.path.join(folder_label, file_label)))
        img_mask = Image.open(os.path.join(folder_mask, mask_filename))

        # 调整mask的尺寸与result图像一致
        img_mask_resized = img_mask.resize(img_result.shape[1::-1], Image.NEAREST)  # 使用NEAREST方式保留mask的二值性
        img_mask_resized = np.array(img_mask_resized)

        # 提取前景部分 (mask值为1的区域为前景)
        foreground_result = img_result * img_mask_resized
        foreground_label = img_label * img_mask_resized

        # 转为Tensor进行MSE计算
        foreground_result_tensor = torch.tensor(foreground_result, dtype=torch.float32)
        foreground_label_tensor = torch.tensor(foreground_label, dtype=torch.float32)

        # 计算MSE Loss (仅计算前景部分)
        mse_loss = F.mse_loss(foreground_result_tensor, foreground_label_tensor)
        mse_total += mse_loss.item()
        count += 1

        print(f"文件 {file_result} 和 {file_label} 的前景MSE Loss: {mse_loss.item()}")

    # 计算平均MSE Loss
    if count > 0:
        average_mse_loss = mse_total / count
        print(f"所有图像的平均前景MSE Loss: {average_mse_loss}")
    else:
        print("没有有效的配对数据进行MSE计算。")
