# pip install image-reward
import ImageReward as RM
import os
from tqdm import tqdm
model = RM.load(name=f"/data1/JM/code/BrushNet/pretrain_model/ImageReward.pt", med_config=f'/data1/JM/code/BrushNet/pretrain_model/med_config.json')

image_list = sorted(os.listdir("/data1/JM/code/IP-Adapter-main/result/base_clip_unet/result"))

rewards_list = []
for image_name in tqdm(image_list):
    image_path = os.path.join("/data1/JM/code/IP-Adapter-main/result/base_clip_unet/result/", image_name)
    prompt_path = os.path.join("/data1/JM/code/IP-Adapter-main/dataset_test/HFlickr_testdata_300/text_from_blip2", image_name.split('.')[0] + '.txt')
    with open(prompt_path, 'r') as f:
        prompt = f.readlines()[0]
    rewards = model.score(prompt, 
                        [image_path])
    print(rewards)
    rewards_list.append(rewards)
print(sum(rewards_list) / len(rewards_list))


# unfinetune unet:
# IR 0.319
# AS 5.825
# MSE Loss: 22.67

# finetune unet:
# IR 0.254
# AS 4.733
# MSE Loss: 20.07