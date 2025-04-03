# Convolutional neural network (two convolutional layers)
import torch
import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            #使用两个3*3卷积核替代一个5*5
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),   #第一层的输入通道必须和输入数据的数据通道一致 如rgb图像的通道是3
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6), #归一化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(5 * 4 * 16, num_classes)  # 32 是卷积最后输出的通道数d， 7 是卷积最后输出的w 和 h
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
