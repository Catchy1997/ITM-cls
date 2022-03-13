import os
import argparse

def get_argparse():
    parser = argparse.ArgumentParser(description='Mydataset classification')
    parser.add_argument('--model_name', default='resnet50', help='')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size for training')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--ngpu', default=1, type=int, help='gpus')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--resume_net', default=None, help='resume net for retraining')
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    parser.add_argument('--output_dir', default='./weights/', help='Location to save checkpoint models')
    parser.add_argument('--gradient_accumulation_steps', default=2)
    return parser.parse_args()


pretrained_dict = {
    "vgg16"    : "./weights/pretrained_models/vgg16_reducedfc.pth",
    # "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    # "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth", 
    # "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth", 
    "resnet50" : "./weights/pretrained_models/resnet50-19c8e357.pth",
    "resnet101" : "./weights/pretrained_models/resnet101-5d3b4d8f.pth",
    # "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth", 
    # "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth", 
    "resnet152" : "./weights/pretrained_models/resnet152-b121ed2d.pth",
    "densenet121" : "./weights/pretrained_models/densenet121-a639ec97.pth",
    "densenet161" : "./weights/pretrained_models/densenet161-17b70270.pth",
    "senet154" : "./weights/pretrained_models/senet154-c7b49a05.pth",
    "se_resnet101" : "./weights/pretrained_models/se_resnet101-7e38fcc6.pth",
    "inceptionv4" : "./weights/pretrained_models/inceptionv4-8e4777a0.pth",
    "se_resnext101_32x4d" : "./pretrained_models/weights/se_resnext101_32x4d-3b2fe3d8.pth",
    "xception" : "./weights/pretrained_models/xception-b0b7af25.pth",
    "inceptionresnetv2" : "./weights/pretrained_models/inceptionresnetv2-520b38e4.pth"
}

entertain = {
    'Trainroot' : "./data/cifar/train",
    'Valroot' : "./data/cifar/test",
    'pretrained_dict' : pretrained_dict,
    'bgr_means' : (104, 117, 123),
    'img_hw' : (224, 224),
    'start_epoch' : 0,
    'end_epoch' : 1,
    'epoch_step' : [0, 2, 4],
    'output_dir' : './weights/',
    'num_classes' : 2,
    'Testroot' : "./data/cifar/test"
}
