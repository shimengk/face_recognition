import os

import cv2
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from face_model import facenet


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
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, feat_num, bias=False),
        )
        self.arc_softmax = Arcsoftmax(feat_num, 25)

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
    cosa = torch.matmul(face1, face2.t())
    return cosa


class FaceRecognition:
    device = 'cuda'

    net_type = 121
    feat_num = 512

    pt_name = f'dense{net_type}_f{feat_num}_last.pt'
    feats_dir = f'res/feats'

    # feats_dir = f'feats{feat_num}_den{net_type}'

    def __init__(self):
        self.device = 'cuda'
        model = facenet.Facenet(backbone='inception_resnetv1', mode="predict")
        model.load_state_dict(torch.load(r'face_model/facenet_inception_resnetv1.pth', map_location=self.device),
                              strict=False)
        self.net = model.eval()
        self.net.to(self.device)
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        print(self.net)

    def get_feature(self, img):
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(np.asarray(img).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
                torch.FloatTensor).to(self.device)

            return self.net(photo_1)

    def __call__(self, img):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        input_person = torch.from_numpy(
            np.expand_dims(np.transpose(np.asarray(img).astype(np.float64) / 255, (2, 0, 1)), 0)).type(
            torch.FloatTensor).to(self.device)

        input_feature = self.net(input_person)
        # 与特征库对比
        sim_dict = {}
        threshold = 0.8

        for feat in os.listdir(self.feats_dir):
            pt_path = os.path.join(self.feats_dir, feat)
            load_feat = torch.load(pt_path).to(self.device)

            # print(compare(input_feature, load_feat))
            siam = torch.linalg.norm(input_feature-load_feat, axis=1)
            # sim_dict[feat.split('.')[0]] = siam.item()
            sim_dict[feat.split('.')[0]] = siam[siam < threshold].numel()
        sorted_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
        print(sorted_dict)
        max_key = next(iter(sorted_dict))  # 获取相似度最大键
        max_value = sorted_dict[max_key]  # 获取键对应的值
        return max_key, max_value
