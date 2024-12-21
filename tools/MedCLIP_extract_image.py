# run：HF_ENDPOINT=https://hf-mirror.com python MedCLIP_extract_image.py
# reference：https://github.com/RyanWangZf/MedCLIP
# install nltk：pip install -U nltk
# pip install transformers>=4.23.1,<=4.24.0
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel
from medclip import MedCLIPProcessor
import torch
from PIL import Image
import json
import numpy as np
from tqdm import tqdm

# load MedCLIP-ResNet50
# model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
# model.from_pretrained()

# load MedCLIP-ViT
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

dataset_name = 'mimic_cxr-512'  # iu_xray-finetune

json_file_path = f'/dataset/{dataset_name}/annotation.json'
image_dir = f'/dataset/{dataset_name}/images/'
image_feat_path = f'/dataset/{dataset_name}/MedCLIP_image.npz'
print('annotation path:', json_file_path)
print('iamge path:', image_dir)
print('image feats save path:', image_feat_path)

with open(json_file_path, 'r') as f:
    json_data = json.load(f)
image_path = []
for split in ['train','val','test']:
    data = json_data[split]
    for i in data:
        image_path.append(i['image_path'])

image_feat = {}
for data in tqdm(image_path):
    with torch.no_grad():
        if dataset_name == 'iu_xray-finetune':
            input0 = processor(images=Image.open(image_dir + data[0]), text=[""], return_tensors="pt").to(device)
            input1 = processor(images=Image.open(image_dir + data[1]), text=[""], return_tensors="pt").to(device)
            output0 = model(**input0)#.squeeze().cpu().numpy()
            img_embeds0 = output0['img_embeds'].cpu().detach().numpy()
            image_feat.update({data[0]:img_embeds0})
            output1 = model(**input1)#.squeeze().cpu().numpy()
            img_embeds1 = output1['img_embeds'].cpu().detach().numpy()
            image_feat.update({data[1]: img_embeds1})
        if dataset_name == 'mimic_cxr-512':
            image = processor(images=Image.open(image_dir + data[0]), text=[""], return_tensors="pt").to(device)
            image_feature = model(**image)  # .squeeze().cpu().numpy()
            image_feature = image_feature['img_embeds'].cpu().detach().numpy()
            image_feat.update({data[0]: image_feature})

np.savez_compressed(image_feat_path, feature=image_feat)
