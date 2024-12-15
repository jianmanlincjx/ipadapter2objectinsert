import os
import numpy as np
from PIL import Image
import cv2

def find_object_bbox(image):
    """
    使用 OpenCV 找到图像中物体的边界框（假设物体区域与背景有明显对比）
    """
    # 转换为灰度图
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    # 使用阈值将物体区域与背景分开
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓，返回最大轮廓的边界框
    if contours:
        contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contour)
        return (x, y, x + w, y + h)  # 返回物体的边界框
    else:
        return None

def crop_to_object(image_path, output_path, padding=10):
    """
    对图像进行物体裁剪，尺寸不固定，尽量保留物体的完整区域。
    可以通过设置 padding 控制裁剪框外边缘的空白区域
    """
    try:
        with Image.open(image_path) as img:
            # 找到物体的边界框
            bbox = find_object_bbox(img)
            
            if bbox:
                # 获取物体边界框
                x1, y1, x2, y2 = bbox
                object_width = x2 - x1
                object_height = y2 - y1
                
                # 可选：增加 padding 留空白区域（默认为10像素）
                x1 = max(x1 - padding, 0)
                y1 = max(y1 - padding, 0)
                x2 = min(x2 + padding, img.width)
                y2 = min(y2 + padding, img.height)
                
                # 计算裁剪区域的宽高
                crop_width = x2 - x1
                crop_height = y2 - y1

                # 裁剪图像
                img_cropped = img.crop((x1, y1, x2, y2))
                img_cropped.save(output_path)
                print(f"Saved cropped image: {output_path}")
            else:
                print(f"No object found in {image_path}, skipping.")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def process_images_in_folder(input_folder, output_folder, padding=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图像文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            output_path = os.path.join(output_folder, filename)
            crop_to_object(file_path, output_path, padding)

if __name__ == "__main__":
    input_folder = "/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/object"  # 替换为你的输入文件夹路径
    output_folder = "/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/object_"  # 替换为你的输出文件夹路径

    process_images_in_folder(input_folder, output_folder)
