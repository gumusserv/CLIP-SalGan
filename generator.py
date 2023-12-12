import torch.nn as nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, text_embedding_dim=256):
        super(Generator, self).__init__()

        # 图像处理层
        self.conv_vgg1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_vgg2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_scratch = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )


        


        self.sigmoid = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 卷积层用于调整文本特征的通道数
        self.text_feature_conv = nn.Conv2d(512, 64, kernel_size=1)
        # 卷积层用于将融合后的特征通道数调整为64
        self.fusion_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)



    def forward(self, x, text_features):
        # 图像处理路径
        x = self.conv_vgg1(x)
        x = self.maxpool(x)
        x = self.conv_vgg2(x)
        x = self.upsample(x)
        image_features = self.conv_scratch(x).float()

        
    

        # 调整文本特征的形状并通过卷积层
        text_features = text_features.view(text_features.size(0), 512, 1, 1).float()
        text_features = self.text_feature_conv(text_features)
        
        # 通过广播扩展文本特征
        text_features = text_features.expand(-1, -1, 256, 256)
        # print(text_features[0][1])
        # print(text_features[0])

        # 拼接两种特征
        fused_features = torch.cat([image_features, text_features], dim=1)

        # 通过卷积调整通道数
        fused_features = self.fusion_conv(fused_features)

        # 通过Sigmoid层
        output = self.sigmoid(fused_features)
        return output



