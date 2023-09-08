import os

import cv2
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)))
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self, net_type, feat_num):
        super(FaceNet, self).__init__()
        if net_type == 121:
            self.sub_net = nn.Sequential(
                models.densenet121()
            )
        elif net_type == 161:
            self.sub_net = nn.Sequential(
                models.densenet161()
            )
        elif net_type == 201:
            self.sub_net = nn.Sequential(
                models.densenet201()
            )
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, feat_num, bias=False),
        )
        self.arc_softmax = Arcsoftmax(feat_num, 22)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.t())
    return cosa


class FaceRecognition:
    device = 'cuda'

    net_type = 121
    feat_num = 512

    pt_name = f'dense121_f512_last.pt'
    feats_dir = f'res/feats_add'

    # feats_dir = f'feats{feat_num}_den{net_type}'

    def __init__(self):
        self.net = FaceNet(self.net_type, self.feat_num).to(self.device)
        self.net.load_state_dict(torch.load(self.pt_name))
        self.net.eval()
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def get_feature(self, img):
        with torch.no_grad():
            input_person = self.tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")).to(self.device)
            return self.net.encode(input_person[None, ...])

    def __call__(self, img):
        input_person = self.tf(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGB")).to(self.device)
        input_feature = self.net.encode(input_person[None, ...])
        # 与特征库对比
        sim_dict = {}
        threshold = 0.7

        for feat in os.listdir(self.feats_dir):
            pt_path = os.path.join(self.feats_dir, feat)
            load_feat = torch.load(pt_path)

            siam = compare(input_feature, load_feat)
            sim_dict[feat.split('.')[0]] = torch.sum(siam > threshold).item()

        sorted_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
        # print(sorted_dict)
        max_key = next(iter(sorted_dict))  # 获取第一个键
        max_value = sorted_dict[max_key]   # 获取第一个键对应的值

        return max_key, max_value
