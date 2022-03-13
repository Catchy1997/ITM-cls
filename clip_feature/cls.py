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
from torchvision.datasets import CIFAR100 #######
from sklearn.linear_model import LogisticRegression #######
print("Torch version:", torch.__version__)
import subprocess
CUDA_version = [s for s in subprocess.check_output(["nvcc", "--version"]).decode("UTF-8").split(", ") if s.startswith("release")][0].split(" ")[-1]
print("CUDA version:", CUDA_version)

import clip
print('available_models:  ' + ','.join(clip.available_models()))
# # RN50,RN101,RN50x4,RN50x16,RN50x64,ViT-B/32,ViT-B/16,ViT-L/14

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_features(model, dataset):
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size=100)):
            features = model.encode_image(images.to(device))

            all_features.append(features)
            all_labels.append(labels)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()


def load_pretrian_model(model_path):
    model, preprocess = clip.load(model_path, device=device)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess


def train(img_path):
    # 加载模型
    model, preprocess = load_pretrian_model('ViT-B/32')
    # input_resolution = model.visual.input_resolution
    # context_length = model.context_length
    # vocab_size = model.vocab_size
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print("Input resolution:", input_resolution)
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)

    # 加载数据集
    # cifar100 = CIFAR100(img_path, download=False, train=False)
    train = CIFAR100(img_path, download=False, train=True, transform=preprocess)
    test = CIFAR100(img_path, download=False, train=False, transform=preprocess)

    # 提取特征
    # train_features, train_labels = get_features(model, train)
    test_features, test_labels = get_features(model, test)

    # print(train_features.shape, train_labels.shape)
    print(test_features.shape, test_labels.shape)

    # # 分类模型
    # classifier = LogisticRegression(random_state=0, C=0.316, max_iter=10, verbose=1)
    # classifier.fit(train_features, train_labels)

    # # 验证分类模型
    # predictions = classifier.predict(test_features)
    # accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
    # print(f"Accuracy = {accuracy:.3f}")


if __name__ == '__main__':
    img_path = os.path.expanduser("./images")
    train(img_path)