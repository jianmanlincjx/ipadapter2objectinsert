import torch
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
processor = BlipProcessor.from_pretrained("pretrain_models/models--Salesforce--blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("pretrain_models/models--Salesforce--blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

image_dir = '/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/target'
save_dir = '/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/text_from_blip2'
os.makedirs(save_dir, exist_ok=True)
image_list = sorted(os.listdir(image_dir))


for image in tqdm(image_list):
    image_path = os.path.join(image_dir, image)
    save_path = os.path.join(save_dir, image).replace('png', 'txt')
    raw_image = Image.open(image_path).convert('RGB')
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)
    out = model.generate(**inputs)
    result = processor.decode(out[0], skip_special_tokens=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(result)