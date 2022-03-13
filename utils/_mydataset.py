# -*- coding:utf-8 -*-
# ! pip install ftfy regex tqdm
# ! pip install git+https://github.com/openai/CLIP.git
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import clip
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_classes(directory):
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def default_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB") # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1


def my_loader(path):
    img = torch.randn(1, 512)
    text = torch.randn(1, 512)
    return img, text


class EntertainDataset(Dataset):
    def __init__(self, root, loader=default_loader, transform=None):
        imgs = []
        for label in os.listdir(root):
            for imname in os.listdir(os.path.join(root, label)):
                imfile = os.path.join(root, label, imname)
                imgs.append((imfile, label))

        self.classes, self.class_to_idx = find_classes(root)
        if self.class_to_idx is None:
            self.classes, self.class_to_idx = find_classes(root)
        elif not self.class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")
        
        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img, text = self.loader(fn)

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, text, self.class_to_idx[label]

    def __len__(self):
        return len(self.imgs)


def gene_dataloder(args, data_root, loader=default_loader):
    """
        # torchvision 的 ImageFolder 能以目录名为标签,对数据集做划分
        # self.classes 保存类名的list ['cat', 'dog']; self.class_to_idx类名对应的索引 {'cat': 0, 'dog': 1}; self.imgs 保存(img_path,class)tuple的list [('./data/cifar/train/cat/cat1.png', 0),]
        # return <class 'torchvision.datasets.folder.ImageFolder'> [(Tensor([3,224,224]), class)]

        # x = torch.linspace(1, 10, 10)
        # y = torch.linspace(10, 1, 10)
        # dataset = torch.utils.data.TensorDataset(x, y)
    """
    # dataset = torchvision.datasets.ImageFolder(root=train_root, transform=transform_dic['train']) 
    dataset = EntertainDataset(root=data_root, loader=loader)
    """
        for step, (batch_x, batch_y) in enumerate(loader):
            print("step:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
        # batch_x:tensor([[[[ ]]]])
        # batch_y:tensor([1, 1, 1, 1]
    """
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return loader
    