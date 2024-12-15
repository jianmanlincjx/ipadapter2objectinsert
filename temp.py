# # import os

# # def rename_files_sorted(directory, substrings_to_remove):
# #     # 获取目录下所有文件
# #     files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
# #     # 去除文件名中的特定字符串，得到一个新的排序列表
# #     def remove_substrings_from_filename(file_name):
# #         for substring in substrings_to_remove:
# #             file_name = file_name.replace(substring, "")
# #         return file_name

# #     # 按去除指定字符串后的文件名排序
# #     files.sort(key=lambda f: remove_substrings_from_filename(f).lower())

# #     # 遍历排序后的文件，进行重命名
# #     for index, file in enumerate(files, start=0):
# #         old_path = os.path.join(directory, file)
# #         # 获取文件扩展名
# #         file_extension = os.path.splitext(file)[1]
# #         # 生成新文件名，保持6位数并补零
# #         new_name = f"{index:06}{file_extension}"
# #         new_path = os.path.join(directory, new_name)
        
# #         # 重命名文件
# #         try:
# #             os.rename(old_path, new_path)
# #             print(f"Renamed: {old_path} -> {new_path}")
# #         except Exception as e:
# #             print(f"Failed to rename {old_path} to {new_path}: {e}")

# # if __name__ == "__main__":
# #     folder_path = input("Enter the directory path: ")  # 用户输入目录路径
# #     substrings = input("Enter substrings to remove (comma separated): ").split(",")  # 输入要去除的字符串，以逗号分隔

# #     # 去除字符串前后的空格
# #     substrings = [s.strip() for s in substrings]

# #     rename_files_sorted(folder_path, substrings)


# import os
# from PIL import Image

# # 设置文件夹路径
# input_dir = '/data1/JM/code/IP-Adapter-main/result/base_clip_unet/test'
# output_dir = '/data1/JM/code/IP-Adapter-main/result/base_clip_unet/label'

# # 确保输出目录存在
# os.makedirs(output_dir, exist_ok=True)

# # 获取所有PNG文件
# files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

# # 处理每个PNG文件
# for file in files:
#     # 打开图像
#     img = Image.open(os.path.join(input_dir, file))

#     # 获取图像尺寸
#     width, height = img.size

#     # 假设每个拼接的图片有三部分，中间部分宽度为1/3的总宽度
#     left = 2 * width // 3
#     right = 3 * width // 3
#     top = 0
#     bottom = height

#     # 裁剪出中间的部分
#     cropped_img = img.crop((left, top, right, bottom))

#     # 保存裁剪后的图片
#     cropped_img.save(os.path.join(output_dir, file))

# print("图片处理完成！")

from PIL import Image, ImageFilter
import numpy as np
from controlnet_aux import LineartDetector
import scipy.ndimage

# 加载Lineart检测器
processor_lineart = LineartDetector.from_pretrained("/data1/JM/code/ControlNet/pretrain_model/models--lllyasviel--Annotators")

# 加载图像
image = Image.open('/data1/JM/code/IP-Adapter-main/image.png')
image_label = Image.open('/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/target_processed/0000327.png')

# 定义噪声过滤函数

# 1. 原图不做处理
image_lineart = processor_lineart(image)
image_lineart.save('./lineart_original_lineart.png')

image_label_lineart = processor_lineart(image_label)
image_label_lineart.save('./image_label_lineart.png')

# 2. 均值滤波 (使用scipy.ndimage)
image_np = np.array(image)  # 转为numpy数组
image_mean_filtered = scipy.ndimage.uniform_filter(image_np, size=3)  # 3x3均值滤波
image_mean_filtered_pil = Image.fromarray(image_mean_filtered)
image_lineart_mean = processor_lineart(image_mean_filtered_pil)
image_lineart_mean.save('./lineart_mean_filtered_lineart.png')

# 3. 中值滤波
image_median_filtered = image.filter(ImageFilter.MedianFilter(size=3))  # 中值滤波
image_lineart_median = processor_lineart(image_median_filtered)
image_lineart_median.save('./lineart_median_filtered_lineart.png')

# 4. 高斯滤波
image_gaussian_filtered = image.filter(ImageFilter.GaussianBlur(radius=2))  # 高斯滤波
image_lineart_gaussian = processor_lineart(image_gaussian_filtered)
image_lineart_gaussian.save('./lineart_gaussian_filtered_lineart.png')

# 5. 双边滤波
import cv2
image_cv = np.array(image)  # 转为OpenCV格式
image_bilateral_filtered = cv2.bilateralFilter(image_cv, d=9, sigmaColor=75, sigmaSpace=75)  # 双边滤波
image_bilateral_filtered_pil = Image.fromarray(image_bilateral_filtered)
image_lineart_bilateral = processor_lineart(image_bilateral_filtered_pil)
image_lineart_bilateral.save('./lineart_bilateral_filtered_lineart.png')

print("所有图像已处理并保存。")
