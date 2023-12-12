import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # 第一个和第二个Conv-Scratch层
        self.conv_scratch = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        # 一个max Pooling层
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 最后的全连接层
        # 假设输入图像的大小为256x256，如果不同请调整
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 128 * 128, 1),  # 根据输入图像的大小调整
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("判别器: ---------------------------------------")
        # print(x.shape)
        x = self.conv_scratch(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x


