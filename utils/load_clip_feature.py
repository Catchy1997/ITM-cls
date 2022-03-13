def my_loader(img_path):
    # 加载模型
    model, preprocess = clip.load('ViT-B/32', device=device)
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)

    # 加载数据集
    dataset = EntertainDataset(root=img_path, transform=transform_dict['test'])

    # 提取特征
    img_features = []
    with torch.no_grad():
        for images, _ in DataLoader(dataset, batch_size=100):
            img_features.append(model.encode_image(images.to(device)))

    return torch.cat(img_features)
    # return torch.cat(all_features).cpu().numpy()

if __name__ == '__main__':
    fe, l = my_loader('/content/drive/MyDrive/pytorch_cls/data/cifar/test')
    print(fe.shape, l.shape)
    print(fe[0])
    print(l[0])
