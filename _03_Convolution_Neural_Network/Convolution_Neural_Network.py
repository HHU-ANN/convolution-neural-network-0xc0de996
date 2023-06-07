# 在该文件NeuralNetwork类中定义你的模型 
# 在自己电脑上训练好模型，保存参数，在这里读取模型参数（不要使用JIT读取），在main中返回读取了模型参数的模型

import os

os.system("sudo pip3 install torch")
os.system("sudo pip3 install torchvision")

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
    

class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3*32*32, hidden_size=128, output_size=10):
        super(NeuralNetwork, self).__init__() # 调用父类构造函数，完成初始化
        self.fc1 = nn.Linear(input_size, hidden_size)  # nn.Linear是一个线性层，他将维度从input_size隐射到hidden_size
        self.relu = nn.ReLU() # 激活函数
        self.fc2 = nn.Linear(hidden_size, output_size) # 将维度从hidden_size映射到output_size

    def forward(self, x):
        # 定义计算的流程
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def read_data():
    # 这里可自行修改数据预处理，batch大小也可自行调整
    # 保持本地训练的数据读取和这里一致
    dataset_train = torchvision.datasets.CIFAR10(root='../data/exp03', train=True, download=True, transform=torchvision.transforms.ToTensor())
    dataset_val = torchvision.datasets.CIFAR10(root='../data/exp03', train=False, download=False, transform=torchvision.transforms.ToTensor())
    data_loader_train = DataLoader(dataset=dataset_train, batch_size=256, shuffle=True)
    data_loader_val = DataLoader(dataset=dataset_val, batch_size=256, shuffle=False)
    return dataset_train, dataset_val, data_loader_train, data_loader_val

def main():
    model = NeuralNetwork() # 若有参数则传入参数
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    model.load_state_dict(torch.load(parent_dir + '/pth/model.pth'))
    return model
    