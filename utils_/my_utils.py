import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CenterLoss(nn.Module):
    def __init__(self, cls_num, feat_num):
        super().__init__()
        self.cls_num = cls_num
        self.center = nn.Parameter(torch.randn(cls_num, feat_num))

    def forward(self, xs, ys):
        xs = F.normalize(xs)
        center_exp = self.center.index_select(dim=0, index=ys.long())

        count = torch.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)

        count_exp = count.index_select(dim=0, index=ys.long())

        center_loss = torch.sum(torch.div(torch.sqrt(torch.sum(torch.pow(xs - center_exp, 2), dim=1)), count_exp))

        return center_loss


class ArcSoftMax(nn.Module):
    def __init__(self, feature_dim=2, cls_dim=10):
        super().__init__()
        # x[n,v] · w[v,c]
        self.W = nn.Parameter(torch.randn((feature_dim, cls_dim)))

    def forward(self, feature, s=10, m=1):
        # 二范数归一化，相当于向量除以模  x → x / ||x||   w → w / ||w||
        x = F.normalize(feature, dim=1)
        w = F.normalize(self.W, dim=0)
        # cosθ = 二范数归一化后的 x·w = (x / ||x||)(w / ||w||)
        # /10：防止梯度爆炸，要在后边乘回来
        cos_theta = torch.matmul(x, w) / 15
        # 反余弦求角度
        theta = torch.acos(cos_theta)

        top = torch.exp(s * torch.cos(theta + m))
        down2 = torch.sum(torch.exp(s * cos_theta), dim=1, keepdim=True) - torch.exp(s * cos_theta)

        arcsoftmax = torch.log(top / (top + down2))

        return arcsoftmax


def IOU(arr1, arr2):
    # 求两个框的面积
    area1 = (arr1[2] - arr1[0]) * (arr1[3] - arr1[1])
    area2 = (arr2[2] - arr2[0]) * (arr2[3] - arr2[1])
    # 比较两个x1，y1的大小，取最大值
    x1_max = np.maximum(arr1[0], arr2[0])
    y1_max = np.maximum(arr1[1], arr2[1])
    # 比较两个x2，y2的大小，取最小值
    x2_min = np.minimum(arr1[2], arr2[2])
    y2_min = np.minimum(arr1[3], arr2[3])
    # 交集
    w = np.maximum(x2_min - x1_max, 0)
    h = np.maximum(y2_min - y1_max, 0)
    wh = w * h
    # 交集 / 并集
    iou = wh / (area1 + area2 - wh)
    return iou


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    return np.sqrt(np.power((x1 - x2), 2) + np.power((y1 - y2), 2))


if __name__ == '__main__':
    input_data = torch.randn(3, 2)
    arc_soft_max = ArcSoftMax()
    out = arc_soft_max(input_data)
    print(out)
