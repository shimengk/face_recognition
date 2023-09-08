import random

import cv2
import torch
import torchvision.models as models
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import *


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


loss_fn = nn.NLLLoss()
device = 'cuda'


def train(net_type, feat_num):
    writer = SummaryWriter('logs')

    train_path = r"D:\data\face_train2"
    test_path = r"D:\data\face_train2"

    last_pt_name = f"dense{net_type}_f{feat_num}_last.pt"
    best_pt_name = f"dense{net_type}_f{feat_num}_best.pt"
    # 训练过程
    net = FaceNet(net_type, feat_num).to(device)

    optimizer = optim.Adam(net.parameters())
    # optimizer = optim.SGD(net.parameters(),lr=0.0001,momentum=0.9)

    dataset = MyDataset(train_path)
    dataloader = DataLoader(dataset=dataset, batch_size=20, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(dataset=MyDataset(test_path), batch_size=8, shuffle=True)
    max_acc = 0
    net.train()
    for epoch in range(1000):
        sum_loss = 0.
        for xs, ys in dataloader:
            feature, cls = net(xs.to(device))

            loss = loss_fn(torch.log(cls), ys.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(torch.argmax(cls, dim=1), ys)
        print(str(epoch) + "Loss====>" + str(loss.item()))
        # if epoch%100==0:
        #     torch.save(net.state_dict(), "params/net.pt")
        #     print(str(epoch)+"参数保存成功")
        torch.save(net.state_dict(), last_pt_name)
        print(str(epoch) + "参数保存成功")
        sum_loss +=loss.item()
        writer.add_scalar('loss', loss.item(), epoch)

        with torch.no_grad():
            net.eval()
            sum_acc = 0.
            for input, target in test_dataloader:
                # input, target = input.reshape(-1, 28 * 28).to(device), target.to(device)
                input, target = input.to(device), target.to(device)

                feat, out = net(input)

                # 测试得分
                out = torch.argmax(out, dim=1)
                sum_acc += torch.mean(torch.eq(out, target).float()).item()

            avg_acc = sum_acc / len(test_dataloader)
            print(f'epoch:{epoch}, acc: {avg_acc}')
            writer.add_scalar('acc', avg_acc, epoch)

            if avg_acc >= max_acc:
                torch.save(net.state_dict(), best_pt_name)


def test(net_type, feat_num):
    # 使用
    net = FaceNet(net_type, feat_num).to(device)
    net.load_state_dict(torch.load(f"dense{net_type}_f{feat_num}_last.pt"))
    net.eval()

    test_root = r'D:\data\face_group'
    mode = 2
    sum = 0
    i = 0
    while True:
        if mode == 1:
            # 随机挑选同一个人
            img_dir = random.sample(os.listdir(test_root), 1)[0]
            img_list = [os.path.join(test_root, img_dir, file_name) for file_name in
                        os.listdir(os.path.join(test_root, img_dir))]
            img1, img2 = random.sample(img_list, 2)
        else:
            # 挑选不同两个人
            img_dir1, img_dir2 = random.sample(os.listdir(test_root), 2)
            img_list1 = [os.path.join(test_root, img_dir1, file_name) for file_name in
                         os.listdir(os.path.join(test_root, img_dir1))]
            img_list2 = [os.path.join(test_root, img_dir2, file_name) for file_name in
                         os.listdir(os.path.join(test_root, img_dir2))]

            img1, img2 = random.sample(img_list1, 1)[0], random.sample(img_list2, 1)[0]

        person1 = tf(Image.open(img1).convert("RGB")).to(device)
        person1_feature = net.encode(person1[None, ...])

        person2 = tf(Image.open(img2).convert("RGB")).to(device)
        person2_feature = net.encode(person2[None, ...])

        siam = compare(person1_feature, person2_feature)
        print(f'sim:{siam.item()}')

        cv2.imshow('img1', cv2.imread(img1))
        cv2.imshow('img2', cv2.imread(img2))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        i += 1
        if mode ==1:
            sum = sum+1 if siam > 0.95 else sum
        else:
            sum = sum+1 if siam < 0.85 else sum

        print(f'准确率:{sum / i}')
    cv2.destroyAllWindows()


def signin(person_dir, person_name, net, load_pt_name):
    feature_list = []
    for file_name in os.listdir(person_dir):
        file_path = os.path.join(person_dir, file_name)
        img = tf(Image.open(file_path).convert("RGB")).to(device)
        with torch.no_grad():
            feature = net.encode(img[None, ...])
            feature_list.append(feature)

    # shape: n,1,v
    stack_tensors = torch.stack(feature_list, dim=0).reshape(-1,512)
    # 改动  按所有次数存储
    # mean_feature = torch.mean(stack_tensors, dim=0)

    save_dir = load_pt_name.split('.')[0]
    os.makedirs(save_dir, exist_ok=True)

    torch.save(stack_tensors, f'{save_dir}/{person_name}.pt',)
    print(save_dir)
    print(stack_tensors.shape)


def login(net, root, load_pt_name):
    img_dir = random.sample(os.listdir(root), 1)[0]
    img_list = [os.path.join(root, img_dir, file_name) for file_name in
                os.listdir(os.path.join(root, img_dir))]
    img = random.sample(img_list, 1)[0]

    input_person = tf(Image.open(img).convert("RGB")).to(device)
    input_feature = net.encode(input_person[None, ...])
    # 与特征库对比
    sim_dict = {}
    threshold = 0.85

    for feat in os.listdir(load_pt_name.split('.')[0]):

        key = feat.split('.')[0]

        pt_path = os.path.join(load_pt_name.split('.')[0], feat)
        load_feat = torch.load(pt_path)

        # for f in load_feat:
        siam = compare(input_feature, load_feat)
        sim_dict[key] = torch.sum(siam > threshold)

    # siam = compare(input_feature, load_feat)
    # sim_dict[feat.split('.')[0]] = siam.item()
    sorted_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
    # print(sorted_dict)
    first_key = next(iter(sorted_dict))  # 获取第一个键
    first_value = sorted_dict[first_key]  # 获取第一个键对应的值
    # print(f'你是{img_dir}, 识别结果是{first_key,first_value}')
    print(f'{img_dir}======{first_key}={first_value}')

    res = str(img_dir) == str(first_key)
    # if not res:
    #     cv2.imshow('img', cv2.imread(img))
    #     cv2.waitKey()
    #     cv2.destroyAllWindows()
    return res


def signin_all(load_pt_name, root, net_type, feat_num):
    net = FaceNet(net_type, feat_num).to(device)
    net.load_state_dict(torch.load(load_pt_name))
    net.eval()
    for file_dir in os.listdir(root):
        signin(os.path.join(root, file_dir), file_dir, net, load_pt_name)


def random_login(load_pt_name, root, net_type, feat_num):
    net = FaceNet(net_type, feat_num).to(device)
    net.load_state_dict(torch.load(load_pt_name))
    net.eval()

    right_count = 0
    for i in range(200):
        res = login(net, root, load_pt_name)
        right_count += int(res)
    print(f'精确度{right_count / 200}')


if __name__ == '__main__':

    """
    121 512 测试同一个准确率 23
    121 128 同一个人        33
    """
    net_type = 121
    feat_num = 512
    train(net_type, feat_num)
    # test(net_type, feat_num)
    #
    #
    load_pt_name = f"dense{net_type}_f{feat_num}_last.pt"
    root = r'D:\data\face_group'

    signin_all(load_pt_name, root, net_type, feat_num)
    random_login(load_pt_name, root, net_type, feat_num)
    # 把模型和参数进行打包，以便C++或PYTHON调用
    # x = torch.Tensor(1, 3, 112, 112)
    # traced_script_module = jit.trace(net, x)
    # traced_script_module.save("model.cpt")
