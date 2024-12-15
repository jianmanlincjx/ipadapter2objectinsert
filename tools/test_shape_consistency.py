import os
from PIL import Image
import numpy as np

# 设置输入图像和mask文件夹路径
image_folder = 'dataset_test/HFlickr_testdata_300/target'
mask_folder = 'dataset_test/HFlickr_testdata_300/mask'
output_folder = 'dataset_test/HFlickr_testdata_300/object'

# 如果输出文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 获取图像文件和对应的mask文件
image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

# 遍历所有图像文件
for image_file in image_files:
    # 构建图像和mask的完整路径
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, image_file)  # 假设mask文件名与图像文件名相同
    
    if image_file in mask_files:  # 确保有对应的mask文件
        # 打开图像和mask
        image = Image.open(image_path).convert('RGBA')  # 将图像转为RGBA模式（包括透明度）
        mask = Image.open(mask_path).convert('L')  # 将mask转为灰度图（单通道）

        # 将mask转换为NumPy数组
        mask_array = np.array(mask)

        # 将mask区域为白色的部分提取出来，其他部分变为黑色
        image_array = np.array(image)
        image_array[mask_array == 0] = [0, 0, 0, 255]  # 将背景部分设置为黑色 (RGBA的黑色，最后的255表示不透明)

        # 将处理后的数组转换回图像
        foreground = Image.fromarray(image_array)

        # 保存前景图像
        output_path = os.path.join(output_folder, image_file)
        foreground.save(output_path)
        print(f"Saved foreground image: {output_path}")

print("Foreground extraction completed.")


# import os
# from PIL import Image

# def check_image_shape_consistency(foreground_folder, input_folder, output_folder, mask_folder):
#     # 获取四个文件夹中的所有图像文件
#     fg_files = os.listdir(foreground_folder)
#     input_files = os.listdir(input_folder)
#     output_files = os.listdir(output_folder)
#     mask_files = os.listdir(mask_folder)

#     # 确保四个文件夹中的文件都是.png类型（或其他类型）
#     fg_files = [f for f in fg_files if f.lower().endswith('.png')]
#     input_files = [f for f in input_files if f.lower().endswith('.png')]
#     output_files = [f for f in output_files if f.lower().endswith('.png')]
#     mask_files = [f for f in mask_files if f.lower().endswith('.png')]

#     # 检查文件名配对是否一致
#     common_files = set(fg_files) & set(input_files) & set(output_files) & set(mask_files)
#     print(len(common_files))
#     if not common_files:
#         print("没有找到配对的文件。")
#         return

#     # 遍历每对配对文件
#     idx = 0
#     for filename in sorted(common_files):
#         fg_path = os.path.join(foreground_folder, filename)
#         input_path = os.path.join(input_folder, filename)
#         output_path = os.path.join(output_folder, filename)
#         mask_path = os.path.join(mask_folder, filename)
        
#         # 打开图像文件，获取图像的尺寸
#         fg_img = Image.open(fg_path)
#         input_img = Image.open(input_path)
#         output_img = Image.open(output_path)
#         mask_img = Image.open(mask_path)

#         # 获取图像的尺寸
#         fg_shape = fg_img.size  # (width, height)
#         input_shape = input_img.size
#         output_shape = output_img.size
#         mask_shape = mask_img.size

#         # 检查形状是否一致
#         if fg_shape != input_shape or fg_shape != output_shape or fg_shape != mask_shape:
#             print(f"图像 {filename} 的形状不一致:")
#             print(f"Foreground shape: {fg_shape}, Input shape: {input_shape}, Output shape: {output_shape}, Mask shape: {mask_shape}")
#         else:
#             idx += 1
#             print(f"图像 {filename} 的形状一致: {fg_shape}")
#     print(idx)
# # 传入四个文件夹的路径
# foreground_folder = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/object_processed'
# input_folder = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/source_processed'
# output_folder = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/target_processed'
# mask_folder = '/data1/JM/code/IP-Adapter-main/dataset/MSRA-10K/mask_processed'

# check_image_shape_consistency(foreground_folder, input_folder, output_folder, mask_folder)
