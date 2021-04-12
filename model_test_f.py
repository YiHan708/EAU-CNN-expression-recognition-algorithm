#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torchvision as tv
import torchvision.transforms as transforms
import torch as t
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torch.nn as nn
import math
import array
import numpy as np
import os
# 定义数据的处理方式
AUg1_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((45, 90)),
        #transforms.Resize(256),
        # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
        #transforms.RandomResizedCrop(227),
        # 图像用于翻转
        #transforms.RandomHorizontalFlip(),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差，IMAGEnet计算来的
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((45, 90)),
        #transforms.Resize(227),
        #transforms.CenterCrop(227),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg2_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((45, 220)),
        # 转换成tensor向量
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((45, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg3_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((110, 220)),
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((110, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg4_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((140, 170)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg5_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((120, 220)),
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((120, 220)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg6_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((110, 220)),#(h,w)
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((110, 220)),#(h,w)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg7_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((80, 190)),
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((80, 190)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

AUg8_transforms = {
    'train': transforms.Compose([
        # 将图像进行缩放，缩放为256*256
        transforms.Resize((80, 130)),
        transforms.ToTensor(),
        # 对图像进行归一化操作
        # [0.485, 0.456, 0.406]，RGB通道的均值与标准差
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
    'val': transforms.Compose([
        transforms.Resize((80, 130)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir =r'../data/ronghe_AUgi/'# r'图像文件夹路径'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')#convert('L')

#模型结构 AUg1的特征
class Aug1_B(nn.Module):
    def __init__(self):
        super(Aug1_B,self).__init__()
        self.conv1=nn.Sequential(#input shape(3,45,90)
                            nn.Conv2d(in_channels=3,out_channels=16,
                                      kernel_size=3,stride=1,padding=0),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2))
        self.conv2=nn.Sequential( #input shape(16,88,43)
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out1=nn.Linear(64*3*9,576)
        self.out2 = nn.Linear(576, 129)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x = self.conv3(x)
        x=x.view(x.size(0),-1)#展平
        x=self.out1(x)
        output=self.out2(x)
        return output

class Aug2_B(nn.Module):
    def __init__(self):
        super(Aug2_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out1=nn.Linear(64*3*25,2400)
        self.out2=nn.Linear(2400,800)
        self.out3 = nn.Linear(800,290)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x= self.out1(x)
        x=self.out2(x)
        output = self.out3(x)
        return output

class Aug3_B(nn.Module):
    def __init__(self):
        super(Aug3_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )
        self.out = nn.Linear(64* 5 * 11, 716)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        output = self.out(x)
        return output

class Aug4_B(nn.Module):
    def __init__(self):
        super(Aug4_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.out1 = nn.Linear(64 * 6 * 8, 1200)
        self.out2=nn.Linear(1200,704)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out1(x)
        output = self.out2(x)
        return output

class Aug5_B(nn.Module):
    def __init__(self):
        super(Aug5_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.out1 = nn.Linear(64 * 5 * 11, 1200)
        self.out2= nn.Linear(1200, 782)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out1(x)
        output = self.out2(x)
        return output

class Aug6_B(nn.Module):
    def __init__(self):
        super(Aug6_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4= nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.out1 = nn.Linear(64 * 5 * 11, 1200)
        self.out2 = nn.Linear(1200, 717)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out1(x)
        output = self.out2(x)
        return output

class Aug7_B(nn.Module):
    def __init__(self):
        super(Aug7_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.out1 = nn.Linear(64 * 3 * 10, 900)
        self.out2 = nn.Linear(900, 451)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out1(x)
        output = self.out2(x)
        return output

class Aug8_B(nn.Module):
    def __init__(self):
        super(Aug8_B, self).__init__()
        self.conv1 = nn.Sequential(  # input shape(3,45,90)
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(  # input shape(16,88,43)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out1 = nn.Linear(64 * 8 * 14, 2800)
        self.out2 = nn.Linear(2800, 1200)
        self.out3 = nn.Linear(1200, 307)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.out1(x)
        x = self.out2(x)
        output = self.out3(x)
        return output
#融合模型
class CombineNet(nn.Module):
    def __init__(self):
        super(CombineNet, self).__init__()
        self.AUg1 = Aug1_B()
        self.AUg2 = Aug2_B()
        self.AUg3 = Aug3_B()
        self.AUg4 = Aug4_B()
        self.AUg5 = Aug5_B()
        self.AUg6 = Aug6_B()
        self.AUg7 = Aug7_B()
        self.AUg8 = Aug8_B()
        self.fc1=nn.Linear(4096,2000)
        self.fc2 = nn.Linear(2000, 600)
        self.fc3 = nn.Linear(600, 80)
        self.fc4 = nn.Linear(80, 7)

    def forward(self, i_AUg1, i_AUg2, i_AUg3, i_AUg4, i_AUg5,i_AUg6,i_AUg7,i_AUg8,vc_w):
        output1 = self.AUg1(i_AUg1)
        output2 = self.AUg2(i_AUg2)
        output3 = self.AUg3(i_AUg3)
        output4 = self.AUg4(i_AUg4)
        output5 = self.AUg5(i_AUg5)
        output6 = self.AUg6(i_AUg6)
        output7 = self.AUg7(i_AUg7)
        output8 = self.AUg8(i_AUg8)
        w_c=vc_w
        out=t.cat([output1,output2,output3,output4,output5,output6,output7,output8],dim=1)
        out = out*w_c
        out=self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return out

class my_Data_Set(nn.Module):
    def __init__(self, txt, transform_AUg1=None, transform_AUg2=None,transform_AUg3=None,transform_AUg4=None,
                 transform_AUg5=None,transform_AUg6=None,transform_AUg7=None,transform_AUg8=None,
                 target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        AUg1_images = []
        AUg2_images = []
        AUg3_images = []
        AUg4_images = []
        AUg5_images = []
        AUg6_images = []
        AUg7_images = []
        AUg8_images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            #它的格式应该为 xxx.jpg 0
            #               xxx.jpg 2
            line.strip('\n')
            line.rstrip()
            information = line.split()
            #每个AUg增加图片
            AUg1_images.append('AUg1/'+information[0])
            AUg2_images.append('AUg2/' + information[0])
            AUg3_images.append('AUg3/' + information[0])
            AUg4_images.append('AUg4/' + information[0])
            AUg5_images.append('AUg5/' + information[0])
            AUg6_images.append('AUg6/' + information[0])
            AUg7_images.append('AUg7/' + information[0])
            AUg8_images.append('AUg8/' + information[0])
            # 将标签信息由str类型转换为float类型
            labels.append([int(l) for l in information[1:len(information)]])
        self.AUg1_images = AUg1_images
        self.AUg2_images = AUg2_images
        self.AUg3_images = AUg3_images
        self.AUg4_images = AUg4_images
        self.AUg5_images = AUg5_images
        self.AUg6_images = AUg6_images
        self.AUg7_images = AUg7_images
        self.AUg8_images = AUg8_images
        self.labels = labels
        self.transform_AUg1 = transform_AUg1
        self.transform_AUg2 = transform_AUg2
        self.transform_AUg3 = transform_AUg3
        self.transform_AUg4 = transform_AUg4
        self.transform_AUg5 = transform_AUg5
        self.transform_AUg6 = transform_AUg6
        self.transform_AUg7 = transform_AUg7
        self.transform_AUg8 = transform_AUg8
        self.target_transform = target_transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        AUg1_Name = self.AUg1_images[item]
        AUg2_Name = self.AUg2_images[item]
        AUg3_Name = self.AUg3_images[item]
        AUg4_Name = self.AUg4_images[item]
        AUg5_Name = self.AUg5_images[item]
        AUg6_Name = self.AUg6_images[item]
        AUg7_Name = self.AUg7_images[item]
        AUg8_Name = self.AUg8_images[item]
        label = self.labels[item]
        # 读入图像信息
        AUg1_image = self.loader(AUg1_Name)
        AUg2_image = self.loader(AUg2_Name)
        AUg3_image = self.loader(AUg3_Name)
        AUg4_image = self.loader(AUg4_Name)
        AUg5_image = self.loader(AUg5_Name)
        AUg6_image = self.loader(AUg6_Name)
        AUg7_image = self.loader(AUg7_Name)
        AUg8_image = self.loader(AUg8_Name)
        # 处理图像数据
        if self.transform_AUg1 is not None:
            AUg1_image = self.transform_AUg1(AUg1_image)
            AUg2_image = self.transform_AUg2(AUg2_image)
            AUg3_image = self.transform_AUg3(AUg3_image)
            AUg4_image = self.transform_AUg4(AUg4_image)
            AUg5_image = self.transform_AUg5(AUg5_image)
            AUg6_image = self.transform_AUg6(AUg6_image)
            AUg7_image = self.transform_AUg7(AUg7_image)
            AUg8_image = self.transform_AUg8(AUg8_image)

        # 需要将标签转换为float类型，BCELoss只接受float类型
        label =t.from_numpy(np.array(label) )#torch.FloatTensor(label)
        return AUg1_image,AUg2_image,AUg3_image,AUg4_image,AUg5_image,AUg6_image,AUg7_image ,AUg8_image,label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.AUg1_images)

test_Data = my_Data_Set(r'../data/ronghe_AUgi/ronghe_test.txt', transform_AUg1=AUg1_transforms['val'],transform_AUg2=AUg2_transforms['val'],
                         transform_AUg3=AUg3_transforms['val'],transform_AUg4=AUg4_transforms['val'],transform_AUg5=AUg5_transforms['val'],
                         transform_AUg6=AUg6_transforms['val'],transform_AUg7=AUg7_transforms['val'],transform_AUg8=AUg8_transforms['val'],
                         loader=Load_Image_Information)

#预测函数
def pridict():
    # 是否使用GPU
    use_gpu = t.cuda.is_available()
    #加载模型
    model = t.load('FL_ronghe_9_model.pkl')
    test_DataLoader = DataLoader(test_Data, batch_size=16)
    dataloaders = {'test': test_DataLoader}
    model.cuda()
    #输出的不相等的数据
    EMO0_list=[0,0,0,0,0,0,0]
    EMO1_list = [0,0,0,0,0,0,0]
    EMO2_list = [0,0,0,0,0,0,0]
    EMO3_list = [0,0,0,0,0,0,0]
    EMO4_list = [0,0,0,0,0,0,0]
    EMO5_list = [0,0,0,0,0,0,0]
    EMO6_list = [0,0,0,0,0,0,0]
    batch_num=0
    num_equal=0
    with t.no_grad():
        # 调用模型测试
        model.eval()
        # 依次获取所有图像，参与模型训练或测试
        for data in dataloaders['test']:
            # 获取输入
            AUg1_inputs, AUg2_inputs, AUg3_inputs, AUg4_inputs, AUg5_inputs, AUg6_inputs, AUg7_inputs, AUg8_inputs, labels = data
            labels = labels.squeeze(1)
            batch = AUg1_inputs.size()[0]
            #输入权重值为1
            w_c = t.ones(batch, 4096)
            # 判断是否使用gpu
            if use_gpu:
                AUg1_inputs = AUg1_inputs.cuda()
                AUg2_inputs = AUg2_inputs.cuda()
                AUg3_inputs = AUg3_inputs.cuda()
                AUg4_inputs = AUg4_inputs.cuda()
                AUg5_inputs = AUg5_inputs.cuda()
                AUg6_inputs = AUg6_inputs.cuda()
                AUg7_inputs = AUg7_inputs.cuda()
                AUg8_inputs = AUg8_inputs.cuda()
                labels = labels.cuda()
                w_c=w_c.cuda()
            # 网络前向运行
            outputs = model(AUg1_inputs, AUg2_inputs, AUg3_inputs, AUg4_inputs, AUg5_inputs, AUg6_inputs, AUg7_inputs,
                            AUg8_inputs,w_c)
            #计算准确率
            pred = t.max(outputs, 1)[1]
            n_pred=pred.cpu().numpy()
            n_labels=labels.cpu().numpy()
            for n_l in range(len(n_labels)):
                if n_labels[n_l]==0:
                    EMO0_list[n_pred[n_l]]+=1
                elif n_labels[n_l]==1:
                    EMO1_list[n_pred[n_l]] += 1
                elif n_labels[n_l]==2:
                    EMO2_list[n_pred[n_l]] += 1
                elif n_labels[n_l]==3:
                    EMO3_list[n_pred[n_l]] += 1
                elif n_labels[n_l]==4:
                    EMO4_list[n_pred[n_l]] += 1
                elif n_labels[n_l]==5:
                    EMO5_list[n_pred[n_l]] += 1
                elif n_labels[n_l]==6:
                    EMO6_list[n_pred[n_l]] += 1
            train_correct = (pred == labels).sum()
            num_equal+=train_correct.item()
            batch_num +=len(n_pred)

    print("平均准确率=%f,num_equal=%d,batch_num=%d",num_equal/(batch_num),num_equal,batch_num)
    sum= np.sum(EMO0_list)
    for i in range(len(EMO0_list)):
        print("正常"+str(i)+" 精度:",EMO0_list[i]/sum,end='')
    print("\n")
    sum = np.sum(EMO1_list)
    for i in range(len(EMO1_list)):
        print("愤怒" + str(i) + " 精度:", EMO1_list[i] / sum,end='')
    print("\n")
    sum = np.sum(EMO2_list)
    for i in range(len(EMO2_list)):
        print("厌恶" + str(i) + " 精度:", EMO2_list[i] / sum,end='')
    print("\n")
    sum = np.sum(EMO3_list)
    for i in range(len(EMO3_list)):
        print("害怕" + str(i) + " 精度:", EMO3_list[i] / sum,end='')
    print("\n")
    sum = np.sum(EMO4_list)
    for i in range(len(EMO4_list)):
        print("高兴" + str(i) + " 精度:", EMO4_list[i] / sum,end='')
    print("\n")
    sum = np.sum(EMO5_list)
    for i in range(len(EMO5_list)):
        print("悲伤" + str(i) + " 精度:", EMO5_list[i] / sum,end='')
    print("\n")
    sum = np.sum(EMO6_list)
    for i in range(len(EMO6_list)):
        print("惊讶" + str(i) + " 精度:", EMO6_list[i] / sum,end='')


if __name__ == '__main__':
    pridict()
