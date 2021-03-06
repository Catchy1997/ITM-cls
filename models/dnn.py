import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


""" 定义网络 """
class CLIPNet(nn.Module):
    def __init__(self, num_classes):
        super(CLIPNet, self).__init__()
        self.model, _ = clip.load('ViT-B/32', device='cuda' if torch.cuda.is_available() else 'cpu')
        self.token_embedding = nn.Embedding(49408, 512)
        self.positional_embedding = nn.Parameter(torch.empty(77, 512))
        self.text_projection = nn.Parameter(torch.empty(512, 512))
        nn.init.normal_(self.text_projection, std=self.model.transformer.width ** -0.5)
        self.dtype = self.model.visual.conv1.weight.dtype

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.layer1 = nn.Sequential(nn.Linear(1024, 128), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(True))
        self.fc_cls = nn.Sequential(nn.Linear(32, num_classes))

    def encode_image(self, image):
        return self.model.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection.type(self.dtype)

        return x

    def forward(self, img_input, text_input):
        img = self.encode_image(img_input)
        text = self.encode_text(text_input)
        input = torch.cat((img, text), -1).view(img.shape[0], 1, -1).type(torch.float)
        
        out = self.encoder(input)
        out = torch.sigmoid(input)
        out = self.layer1(out)
        out_margin = self.layer2(out)
        out_cls = self.fc_cls(out_margin)
        return out_cls.view(out_cls.shape[0],-1).contiguous(), out_margin.view(out_cls.shape[0],-1).contiguous()


class Net(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(Net,self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_shape, nhead=8)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.layer1 = nn.Sequential(nn.Linear(in_shape, 128), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(True))
        self.fc_cls = nn.Sequential(nn.Linear(32, num_classes))

    def forward(self, img_input, text_input):
        input = torch.cat((img_input, text_input), -1)
        out = self.encoder(input)
        out = torch.sigmoid(out)
        out = self.layer1(out)
        out_margin = self.layer2(out)
        out_cls = self.fc_cls(out_margin)

        return out_cls.view(out_cls.shape[0],-1).contiguous(), out_margin.view(out_cls.shape[0],-1).contiguous()


class FCNet(nn.Module):
    def __init__(self,n_input,n_hidden,num_classes):
        super(FCNet,self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, num_classes)

    def forward(self, input):
        out = self.hidden1(input)
        out = torch.sigmoid(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)
        return out


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Batch_Net(nn.Module):
    """
    定义了一个简单的三层全连接神经网络，每一层都是线性的 nn.Linear
    增加了一个加快收敛速度的方法——批标准化 nn.BatchNorm1d
    在每层的输出部分添加了激活函数 nn.ReLU(True)  
    """
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

