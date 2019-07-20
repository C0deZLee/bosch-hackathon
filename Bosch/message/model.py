import torch
from torch.utils import data  # 获取迭代数据
from torch.autograd import Variable  # 获取变量
import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import mnist  # 获取数据集
import matplotlib.pyplot as plt
import numpy as np

batch_size = 10

data_path = "./data/"


# 定义网络结构
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x

    def save_model(self, name='clf.pkl'):
        torch.save(self.state_dict(), data_path + name)

    def load_model(self, name='clf.pkl'):
        self.load_state_dict(torch.load(data_path + name))


# 数据
from .dataproc import get_data, increase_data


class MyDataset(Dataset):
    def __init__(self, source_data, transforms_=None, mode="train", shuffle=False):
        self.data, self.label = get_data(source_data)
        print("data num:", len(self.data), "label num:",len(self.label))
        # print(self.data)
        if len(self.data) != len(self.label):
            self.data, self.label = get_data(source_data, True)

        if shuffle:
            idx = [i for i in range(len(self.data))]
            self.data = self.data[idx]
            self.label = self.label[idx]

        t = int(len(self.data) * 0.5)
        if mode == "train":
            self.data = self.data[:t]
            self.label = self.label[:t]
        elif mode == 'test':
            self.data = self.data[t:]
            self.label = self.label[t:]
        self.data = self.data.astype(np.float32)
        self.label = torch.LongTensor(self.label)

        self.transform = transforms_

    def __getitem__(self, index):
        data = self.transform(self.data[index])
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)


# 数据集的预处理
data_tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])
]
)

data_path = r'traindata'
# 获取数据集
train_dataset = MyDataset(data_path, data_tf, "train", True)
test_dataset = MyDataset(data_path, data_tf, "test", True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def train(model, epoch_num=10):
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_count = []
    for epoch in range(epoch_num):
        for i, (x, y) in enumerate(train_loader):
            batch_x = Variable(x)  # torch.Size([128, 1, 28, 28])
            batch_y = Variable(y)  # torch.Size([128])
            # 获取最后输出
            out = model(batch_x)  # torch.Size([128,10])
            # 获取损失
            loss = loss_func(out, batch_y)
            # 使用优化器优化损失
            opt.zero_grad()  # 清空上一步残余更新参数值
            loss.backward()  # 误差反向传播，计算参数更新值
            opt.step()  # 将参数更新值施加到net的parmeters上
            if i % 20 == 0:
                loss_count.append(loss)
                print('{}:\t'.format(i), loss.item())
                # torch.save(model, r'tmp.pkl')
            if i % 100 == 0:
                acc = []
                for a, b in test_loader:
                    test_x = Variable(a)
                    test_y = Variable(b)
                    out = model(test_x)
                    # print('test_out:\t',torch.max(out,1)[1])
                    # print('test_y:\t',test_y)
                    accuracy = torch.max(out, 1)[1].numpy() == test_y.numpy()
                    acc.append(float(accuracy.mean()))
                print('accuracy:\t', sum(acc) / len(acc))

    model.save_model()


def test(model):
    from sklearn.metrics import classification_report
    test_dataset = MyDataset(data_path, data_tf, "all", True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    y_true = []
    y_pred = []
    for a, b in test_loader:
        test_x = Variable(a)
        test_y = Variable(b)
        out = model(test_x)
        y_true += test_y.numpy().tolist()
        y_pred += torch.max(out, 1)[1].numpy().tolist()

    target_names = [str(i) for i in range(10)]
    print(classification_report(y_true, y_pred, target_names=target_names))


model = CNNnet()
try:
    model.load_model()
except:
    print("please train model first")

####################
import torch
from PIL import Image


def predict_one_record(data):
    img = Image.fromarray(np.uint8(data), '1')
    img.save('tmp.png')
    dt = test_dataset.transform(img)
    dt = dt.reshape((1, 1, 28, 28))
    out = model(dt)
    ret = torch.max(out, 1)[1].numpy().tolist()[0]
    return ret

import json
def save_one_record(data, label):
    increase_data(data, int(label))
    retrain_model()


def retrain_model():
    global train_dataset
    train_dataset = MyDataset(data_path, data_tf, "all", True)
    train(model, 1)
    print("train success")


if __name__ == "__main__":
    # train(model)
    test(model)
