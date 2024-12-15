import torch
import os
import open_clip
from tqdm import tqdm
from PIL import Image

def calculate_aesthetic_score(img):
    image = clip_preprocess(img).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        prediction = aesthetic_model(image_features)
    return prediction.cpu().item()

# aesthetic model
aesthetic_model = torch.nn.Linear(768, 1)
aesthetic_model_ckpt_path=os.path.join('/data1/JM/code/BrushNet/pretrain_model',"sa_0_4_vit_l_14_linear.pth")
aesthetic_model.load_state_dict(torch.load(aesthetic_model_ckpt_path))
aesthetic_model.eval()
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')


image_list = sorted(os.listdir("/data1/JM/code/IP-Adapter-main/result/base_clip_unet/result"))

rewards_list = []
for image_name in tqdm(image_list):
    image_path = os.path.join("/data1/JM/code/IP-Adapter-main/result/base_clip_unet/result/", image_name)
    as_score_ = calculate_aesthetic_score(Image.open(image_path))
    rewards_list.append(as_score_)
print(sum(rewards_list) / len(rewards_list))