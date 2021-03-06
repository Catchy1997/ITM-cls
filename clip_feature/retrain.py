# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git

import os, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import PIL
from PIL import Image

import torch
from torch import nn, optim
import torchvision
from torch.utils.data import Dataset, DataLoader
print("Torch version:", torch.__version__)
import subprocess
CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

import clip
print('available_models:  ' + ','.join(clip.available_models()))

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]))
        caption = self.caption[idx]
        return images, caption


def load_data(img_path, batch_size, preprocess):
    df = {'image': [], 'caption':[]}
    catgories = os.listdir(img_path)

    for each in catgories:
        caption = each
        filepath = os.path.join(img_path, each)
        for img in os.listdir(filepath):
            df['image'].append(os.path.join(filepath, img))
            df['caption'].append(caption)

    dataset = image_caption_dataset(df, preprocess)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess


def train(epoch, batch_size, learning_rate, img_path):
    # 加载模型
    model, preprocess = load_pretrian_model('ViT-B/32')

    # 加载数据集
    train_dataloader = load_data(img_path, batch_size, preprocess)

    # 设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for i in range(epoch):
        for batch in train_dataloader:
            list_image, list_txt = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images

            # list_image = list_image.to(device)

            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            if device == "cpu":
                ground_truth = torch.arange(batch_size).long().to(device)
            else:
                # ground_truth = torch.arange(batch_size).half().to(device)
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            # 反向传播
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print('[%d] loss: %.3f' %(i + 1, total_loss))
    torch.save(model, './model/model-finetune.pkl')


if __name__ == '__main__':
    epoch = 50
    batch_size = 24
    learning_rate = 5e-5
    img_path = './data'

    train(epoch, batch_size, learning_rate, img_path)