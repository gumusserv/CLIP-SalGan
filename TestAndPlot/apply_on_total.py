
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from generator import *
from discriminator import *
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from score import *


from PIL import Image
import torchvision.transforms.functional as TF

generator = Generator()
discriminator = Discriminator()

generator.load_state_dict(torch.load('generator_model_final_total.pt', map_location = torch.device('cpu')))
discriminator.load_state_dict(torch.load('discriminator_model_final_total.pt', map_location = torch.device('cpu')))

generator.eval()
discriminator.eval()



import json
import os



# 用于获取每个图片-文本对的图片路径, 文本描述
def get_Data(image_paths, target_paths):


    with open('text.json', 'r') as f:
        text_dic = json.load(f)

    text_descriptions = []

    for path in target_paths:
        path = path[path.rfind('/') + 1:]
        first_nonzero_index = None

        for i in range(len(path)):
            if path[i] != '0':
                first_nonzero_index = i
                break
        if first_nonzero_index != None:
            path = path[first_nonzero_index:]
        
        path = path[:path.find('.') ]
        
        text_descriptions.append(text_dic[path])

    return image_paths, target_paths, text_descriptions






# Define the transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

import torch
import clip
from PIL import Image

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型
model, preprocess = clip.load("ViT-B/32", device=device)

class SaliencyDatasetWithText(Dataset):
    def __init__(self, image_paths, target_paths, text_sequences, transform=None):
        self.image_paths = image_paths
        self.target_paths = target_paths
        
        
        self.text_sequences = []
        for i in range(len(image_paths)):
            # 处理图片
            # image = preprocess(Image.open(image_paths[i])).unsqueeze(0).to(device)
            
            # 处理文本
            text_tokens = clip.tokenize([text_sequences[i]]).to(device)

            with torch.no_grad():
                # 生成图片和文本的特征向量
                # image_features = model.encode_image(image)
                text_features = model.encode_text(text_tokens)

                # 打印或存储特征向量
                # print(text_features.cpu().numpy().shape)
                self.text_sequences.append(text_features)




        # self.text_sequences = text_to_embedding(text_sequences)
        # print(self.text_sequences.shape)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        target = Image.open(self.target_paths[idx]).convert('L')
        text = self.text_sequences[idx]

        if self.transform:
            image = self.transform(image)
            target = self.transform(target)

        # 将文本序列转换为 PyTorch 张量
        text_tensor = torch.tensor(text, dtype=torch.long)

        return image, target, text_tensor
def create_dataloader(data, transform, batch_size=4, shuffle=True):
    image_paths, target_paths, text_descriptions = zip(*data)
    dataset = SaliencyDatasetWithText(list(image_paths), list(target_paths), list(text_descriptions), transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加批次维度
    return image




def resize_saliency_map(saliency_map, original_size):
    # 将PyTorch张量转换为PIL图像
    saliency_map_pil = TF.to_pil_image(saliency_map)
    # 调整大小
    resized_saliency_map = saliency_map_pil.resize(original_size, Image.BILINEAR)
    return resized_saliency_map

# 指定目标目录路径
image_directory_path = 'saliency/image_1800'
target_directory_path = 'saliency/map_1800'


with open('test_data_list_total.json', 'r') as f:
    test_data = json.load(f)

# image_paths_all = ['saliency/image_1800/000000000109_0.png', 'saliency/image_1800/000000000109_2.png', 'saliency/image_1800/000000000109_3.png']
# target_paths_all = ['saliency/map_1800/000000000109_0.png', 'saliency/map_1800/000000000109_2.png', 'saliency/map_1800/000000000109_3.png']
image_paths_all = []
target_paths_all = []
print(len(test_data))



for i in range(len(test_data)):
    if "_2.png" in test_data[i][0]:
        image_paths_all.append(test_data[i][0])

        target_paths_all.append(test_data[i][1])
        image_paths_all.append(test_data[i][0].replace("_2", "_3"))

        target_paths_all.append(test_data[i][1].replace("_2", "_3"))
    




total_score_dic = dict()

total_score_dic['pure'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
total_score_dic['nonsal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
total_score_dic['sal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
total_score_dic['general'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
total_score_dic['total'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}



for i in range(0, len(image_paths_all), 2):
    picture_list = []
    ground_truth = []

    for k in range(2):
    
        image_paths, target_paths, text_descriptions = get_Data([image_paths_all[i + k]], [target_paths_all[i + k]])
        print(image_paths)
        print(target_paths)
        if k == 0:
            ground_truth.append(target_paths[0])
            picture_list.append(target_paths[0])
        ground_truth.append(target_paths[0])
        print(text_descriptions)


        val_data = list(zip(image_paths, target_paths, text_descriptions))

        val_loader = create_dataloader(val_data, transform)

        criterion = nn.BCELoss()
        with torch.no_grad():
            val_loss = 0.0
            val_loss2 = 0.0
            for m, (images, targets, texts_embeddings) in enumerate(val_loader):
                print(image_paths[m])
                # 从文件获取原始图像尺寸
                original_image = Image.open(image_paths[m])
                original_size = original_image.size
                # 只计算生成器的损失
                fake_targets = generator(images, texts_embeddings)
                outputs = discriminator(fake_targets)
                val_loss += criterion(outputs, torch.ones(images.size(0), 1)).item()
                picture = fake_targets.squeeze(0)

                AUC_score = AUC(fake_targets, targets)
                sAUC_score = sAUC(fake_targets, targets)
                CC_score = CC(fake_targets, targets)
                NSS_score = NSS(fake_targets, targets)

                print("AUC Score: {}".format(AUC_score))
                print("sAUC Score: {}".format(sAUC_score))
                print("CC Score: {}".format(CC_score))
                print("NSS Score: {}".format(NSS_score))

                if "_2.png" in image_paths[m]:
                    total_score_dic['nonsal']['AUC'].append(AUC_score)
                    total_score_dic['nonsal']['sAUC'].append(sAUC_score)
                    total_score_dic['nonsal']['CC'].append(CC_score)
                    total_score_dic['nonsal']['NSS'].append(NSS_score)
                    
                elif "_3.png" in image_paths[m]:
                    total_score_dic['sal']['AUC'].append(AUC_score)
                    total_score_dic['sal']['sAUC'].append(sAUC_score)
                    total_score_dic['sal']['CC'].append(CC_score)
                    total_score_dic['sal']['NSS'].append(NSS_score)
                elif "_0.png" in image_paths[m]:
                    total_score_dic['pure']['AUC'].append(AUC_score)
                    total_score_dic['pure']['sAUC'].append(sAUC_score)
                    total_score_dic['pure']['CC'].append(CC_score)
                    total_score_dic['pure']['NSS'].append(NSS_score)
                else:
                    total_score_dic['general']['AUC'].append(AUC_score)
                    total_score_dic['general']['sAUC'].append(sAUC_score)
                    total_score_dic['general']['CC'].append(CC_score)
                    total_score_dic['general']['NSS'].append(NSS_score)
                total_score_dic['total']['AUC'].append(AUC_score)
                total_score_dic['total']['sAUC'].append(sAUC_score)
                total_score_dic['total']['CC'].append(CC_score)
                total_score_dic['total']['NSS'].append(NSS_score)



                picture = resize_saliency_map(picture, original_size)

                


                picture_list.append(picture)
            

                print(val_loss)

    # new_pictures = ground_truth
    # print(picture_list)
    # print(new_pictures)
    # # 创建3x3的子图布局，并在每个子图位置上绘制对应的图片
    # # 从上到下分别是ground truth; sal; nonsal;
    # # 从左到右依次是无文本; 非显著文本; 显著文本
    # fig, axes = plt.subplots(2, 3)

    # # 在第一行展示新增的索引为0、1、2的三张图
    # axes[0, 0].imshow(plt.imread(new_pictures[0]), cmap='gray')
    # axes[0, 1].imshow(plt.imread(new_pictures[1]), cmap='gray')
    # axes[0, 2].imshow(plt.imread(new_pictures[2]), cmap='gray')

    # # 将之前的六张图片分别移动到第二行和第三行
    # axes[1, 0].imshow(picture_list[0], cmap='gray')
    # axes[1, 1].imshow(picture_list[1], cmap='gray')
    # axes[1, 2].imshow(picture_list[2], cmap='gray')
    

    # # 隐藏所有子图的坐标轴
    # for ax in axes.flatten():
    #     ax.axis('off')

    # # 调整子图之间的间距
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # 显示图片
    # plt.show()

# print(sal_score_dic)
# print(nonsal_score_dic)

# with open('total_score_dic.json', 'w') as file:
#     json.dump(total_score_dic, file)
# with open('nonsal_score_dic.json', 'w') as file:
#     json.dump(nonsal_score_dic, file)

