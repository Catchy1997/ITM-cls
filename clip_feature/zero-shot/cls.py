# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git
import os, sys, random, glob
import numpy as np
import PIL
from PIL import Image

import torch
print("Torch version:", torch.__version__)
import subprocess
CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

import clip
print('available_models:  ' + ','.join(clip.available_models()))


def encode(object):
    o = object.lower()
    if("jpg" in o[-5:]) or ("png" in o[-5:]) or ("jpeg" in o[-5:]):
        return model.encode_image(preprocess(Image.open(object)).unsqueeze(0).to("cuda:0"))
    else:
        return model.encode_text(clip.tokenize(object).cuda()).detach().clone()


def load_label(classes):
    if classes=="imagenet":
        import pandas as pd
        categories = pd.read_csv("categories/map_clsloc.txt", sep=" ", header = None)[2]
        for category in range(len(categories)):
            categories[category] = categories[category].replace("_", " ")
    elif classes=="pokemon":
        import pandas as pd
        categories = pd.read_csv("categories/pokemon.txt", sep=".", header=None)[1]
    elif classes=="emojis":
        categories = open("categories/emojis.txt", "r").readlines()
    else:
        categories = classes
    
    return categories


def model_load(model_name, device):
    # 加载模型
    # model, preprocess = clip.load(model_name, device) #ViT-B/32 RN50x16
    _, preprocess = clip.load(model_name, device) #ViT-B/32 RN50x16
    model = torch.load('./model/model-finetune.pkl')
    
    return model, preprocess


if __name__ == '__main__':
    img_pah = '/content/drive/MyDrive/video_cls/zero-shot/1.jpg'
    classes = ['zhenhuanzhuan', 'cat'] #@param ["imagenet", "dog vs cat", "pokemon", "words in the communist manifesto", "other (open this cell and write them into a list of strings)"]
    categories = load_label(classes)
    model_name = 'ViT-B/32' #ViT-B/32 RN50x16

    #加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = model_load(model_name, device)

    #准备输入集
    image = Image.open(img_pah)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(device) #生成文字描述

    #特征编码
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    #选取参数最高的标签
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) #对图像描述和图像特征  
    cosine_similarity = [torch.cosine_similarity(e, image_features).item() for e in text_features]

    #输出结果
    return_raw = False
    if return_raw == False:
        values, indices = similarity[0].topk(1)
        print("\nTop predictions:\n")
        print('classes:{} score:{:.2f}'.format(categories[indices.item()], values.item()))
        # classes:zhenhuanzhuan score:0.58
        
        # classes:qingyunian score:0.53

    else:
        result = cosine_similarity
        print(result)