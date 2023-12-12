import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from Data_Utils import *
# from get_data_part import *
from get_data import *
from generator import *
from discriminator import *
from train import *





# 指定目标目录路径
image_directory_path = 'saliency/image_1800'
target_directory_path = 'saliency/map_1800'

image_paths, target_paths, text_descriptions = get_Data(image_directory_path, target_directory_path)






train_data, val_data, test_data = split_dataset(image_paths, target_paths, text_descriptions)


# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

batch_size = 16
# 创建训练、验证和测试数据加载器
train_loader = create_dataloader(train_data, transform, batch_size=batch_size)
val_loader = create_dataloader(val_data, transform, batch_size=batch_size)
test_loader = create_dataloader(test_data, transform, batch_size=batch_size, shuffle=False)  # 测试集通常不需要打乱





device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

generator = Generator().to(device)
discriminator = Discriminator().to(device)



# Define the loss functions and optimizers
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
num_epochs = 50

# 调用函数进行训练
train_model(train_loader, val_loader, generator, discriminator, criterion, optimizer_G, optimizer_D, device, num_epochs=num_epochs)

# 保存生成器模型
torch.save(generator.state_dict(), 'generator_model_final_total.pt')

# 保存判别器模型（如果需要）
torch.save(discriminator.state_dict(), 'discriminator_model_final_total.pt')













