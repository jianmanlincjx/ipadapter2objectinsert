import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel
import sys
sys.path.append("./dinov2")
import hubconf
import numpy as np
import albumentations as A
from einops import rearrange
import cv2

DINOv2_weight_path = 'pretrain_models/DINOV2/dinov2_vitg14_reg4_pretrain.pth'


def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        #A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,  value=(0,0,0)),
        ])

    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask

class FrozenDinoV2Encoder(nn.Module):
    """
    Uses the DINOv2 encoder for image
    """
    def __init__(self, device="cuda", freeze=True):
        super().__init__()
        dinov2 = hubconf.dinov2_vitg14() 
        state_dict = torch.load(DINOv2_weight_path)
        dinov2.load_state_dict(state_dict, strict=False)
        self.model = dinov2.to(device)
        self.device = device
        if freeze:
            self.freeze()
        self.image_mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.image_std =  torch.tensor([0.229, 0.224, 0.225]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)        
        self.projector = nn.Linear(1536,1024)

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        if isinstance(image,list):
            image = torch.cat(image,0)

        image = (image.to(self.device)  - self.image_mean.to(self.device)) / self.image_std.to(self.device)
        features = self.model.forward_features(image)
        tokens = features["x_norm_patchtokens"]
        image_features  = features["x_norm_clstoken"]
        image_features = image_features.unsqueeze(1)
        hint = torch.cat([image_features,tokens],1) # 8,257,1024
        hint = self.projector(hint)
        return hint
    
    def process_data(self, image):
        return image  / 255 

if __name__ == "__main__":
    DINOV2 = FrozenDinoV2Encoder().cuda()
    x = cv2.imread('/data1/JM/code/AnyDoor-main/masked_ref_image.png')
    x = torch.from_numpy(x).cuda().unsqueeze(0).permute(0, 3, 1, 2)
    y = cv2.imread('/data1/JM/code/AnyDoor-main/masked_ref_image_compose.png')
    y = torch.from_numpy(y).cuda().unsqueeze(0).permute(0, 3, 1, 2)

    x_result = DINOV2(x)
    y_result = DINOV2(y)

    cos_similarity = F.cosine_similarity(x_result.view(1, -1), y_result.view(1, -1), dim=1)

    print("Cosine Similarity:", cos_similarity.item())