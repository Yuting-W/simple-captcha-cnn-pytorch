import cv2
import torch
import torch.nn as nn
from notes.ConvNet import ConvNet
import torchvision.transforms as transforms
# 预处理图片
def paa(file):
    # 输入图片必须是 68*23 如果不是，还需要更多的预处理
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图

    img = img[1:-2, 2:-2]

    ret, img = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)  # 二值化
    # 分离数字
    img1 = img[:, 0:16]
    img2 = img[:, 16:32]
    img3 = img[:, 32:48]
    img4 = img[:, 48:64]

    return [img1, img2, img3, img4]


# 预处理图片
imgs = paa('test.jpg')


model = ConvNet()
model.load_state_dict(torch.load('model/model.pth'))  # 加载训练好的参数
model.eval()  # 切换到评估模式

# 进行预测
with torch.no_grad():  # 禁用梯度计算
    print('预测类别：')
    img_tensor = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1)
    outputs = model(img_tensor)  # 输入图像到模型
    label, predicted = torch.max(outputs, 1)  # 预测类别
    for value in predicted:
        print(f'{value.item()}')

