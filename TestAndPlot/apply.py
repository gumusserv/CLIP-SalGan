
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

generator.load_state_dict(torch.load('generator_model_final_sal.pt', map_location = torch.device('cpu')))
discriminator.load_state_dict(torch.load('discriminator_model_final_sal.pt', map_location = torch.device('cpu')))

generator.eval()
discriminator.eval()

generator2 = Generator()
discriminator2 = Discriminator()

generator2.load_state_dict(torch.load('generator_model_final_nonsal.pt', map_location = torch.device('cpu')))
discriminator2.load_state_dict(torch.load('discriminator_model_final_nonsal.pt', map_location = torch.device('cpu')))

generator2.eval()
discriminator2.eval()

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


with open('test_data_list_sal.json', 'r') as f:
    test_data = json.load(f)

# image_paths_all = ['saliency/image_1800/000000000109_0.png', 'saliency/image_1800/000000000109_2.png', 'saliency/image_1800/000000000109_3.png']
# target_paths_all = ['saliency/map_1800/000000000109_0.png', 'saliency/map_1800/000000000109_2.png', 'saliency/map_1800/000000000109_3.png']
image_paths_all = []
target_paths_all = []
print(len(test_data))



for i in range(0, 90, 2):
    image_paths_all.append(test_data[i][0])
    image_paths_all.append(test_data[i][0].replace("_0", "_2"))
    image_paths_all.append(test_data[i][0].replace("_0", "_3"))

    target_paths_all.append(test_data[i][1])
    target_paths_all.append(test_data[i][1].replace("_0", "_2"))
    target_paths_all.append(test_data[i][1].replace("_0", "_3"))


    


sal_score_dic = dict()
nonsal_score_dic = dict()
sal_score_dic['pure'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
sal_score_dic['nonsal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
sal_score_dic['sal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
nonsal_score_dic['pure'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
nonsal_score_dic['nonsal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}
nonsal_score_dic['sal'] = {"AUC" : [], "sAUC" : [], "CC" : [], "NSS" : []}




# # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
# image_paths = [image_directory_path + "/" + f for f in os.listdir(image_directory_path) if os.path.isfile(os.path.join(image_directory_path, f))]

# image_paths = image_paths[0 : ]

# image_paths_all = [item for item in image_paths if "_3.png" in item or "_0.png" in item ]

        
# # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
# target_paths = [target_directory_path + "/" + f for f in os.listdir(target_directory_path) if os.path.isfile(os.path.join(target_directory_path, f))]

# target_paths = target_paths[0 : ]

# target_paths_all = [item for item in target_paths if "_3.png" in item  or "_0.png" in item]

# # 指定目标目录路径
# image_paths = ['saliency/image_1800/1287704027_3.png']
# target_paths = ['saliency/map_1800/1287704027_3.png']
for i in range(0, 3, 3):
    picture_list = []
    ground_truth = []
    for k in range(3):
        image_paths, target_paths, text_descriptions = get_Data([image_paths_all[i + k]], [target_paths_all[i + k]])
        print(image_paths)
        print(target_paths)
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

                if k == 0:
                    sal_score_dic['pure']['AUC'].append(AUC_score)
                    sal_score_dic['pure']['sAUC'].append(sAUC_score)
                    sal_score_dic['pure']['CC'].append(CC_score)
                    sal_score_dic['pure']['NSS'].append(NSS_score)
                elif k == 1:
                    sal_score_dic['nonsal']['AUC'].append(AUC_score)
                    sal_score_dic['nonsal']['sAUC'].append(sAUC_score)
                    sal_score_dic['nonsal']['CC'].append(CC_score)
                    sal_score_dic['nonsal']['NSS'].append(NSS_score)
                else:
                    sal_score_dic['sal']['AUC'].append(AUC_score)
                    sal_score_dic['sal']['sAUC'].append(sAUC_score)
                    sal_score_dic['sal']['CC'].append(CC_score)
                    sal_score_dic['sal']['NSS'].append(NSS_score)


                picture = resize_saliency_map(picture, original_size)

                fake_targets2 = generator2(images, texts_embeddings)

                AUC_score2 = AUC(fake_targets2, targets)
                sAUC_score2 = sAUC(fake_targets2, targets)
                CC_score2 = CC(fake_targets2, targets)
                NSS_score2 = NSS(fake_targets2, targets)

                print("AUC Score: {}".format(AUC_score2))
                print("sAUC Score: {}".format(sAUC_score2))
                print("CC Score: {}".format(CC_score2))
                print("NSS Score: {}".format(NSS_score2))

                if k == 0:
                    nonsal_score_dic['pure']['AUC'].append(AUC_score2)
                    nonsal_score_dic['pure']['sAUC'].append(sAUC_score2)
                    nonsal_score_dic['pure']['CC'].append(CC_score2)
                    nonsal_score_dic['pure']['NSS'].append(NSS_score2)
                elif k == 1:
                    nonsal_score_dic['nonsal']['AUC'].append(AUC_score2)
                    nonsal_score_dic['nonsal']['sAUC'].append(sAUC_score2)
                    nonsal_score_dic['nonsal']['CC'].append(CC_score2)
                    nonsal_score_dic['nonsal']['NSS'].append(NSS_score2)
                else:
                    nonsal_score_dic['sal']['AUC'].append(AUC_score2)
                    nonsal_score_dic['sal']['sAUC'].append(sAUC_score2)
                    nonsal_score_dic['sal']['CC'].append(CC_score2)
                    nonsal_score_dic['sal']['NSS'].append(NSS_score2)


                outputs2 = discriminator2(fake_targets2)
                val_loss2 += criterion(outputs2, torch.ones(images.size(0), 1)).item()
                picture2 = fake_targets2.squeeze(0)
                picture2 = resize_saliency_map(picture2, original_size)


                picture_list.append(picture)
                picture_list.append(picture2)

                print(val_loss)






    
    # 可视化
    # plt.imshow(picture, cmap='gray')
    # plt.show()
    



    # 创建2x3的子图布局，并在每个子图位置上绘制对应的图片
    # fig, axes = plt.subplots(2, 3)

    # # 在第一行展示索引为0、2、4的三张图
    # axes[0, 0].imshow(picture_list[0], cmap='gray')
    # axes[0, 1].imshow(picture_list[2], cmap='gray')
    # axes[0, 2].imshow(picture_list[4], cmap='gray')
    # axes[0, 0].axis('off')
    # axes[0, 1].axis('off')
    # axes[0, 2].axis('off')

    # # 在第二行展示索引为1、3、5的三张图
    # axes[1, 0].imshow(picture_list[1], cmap='gray')
    # axes[1, 1].imshow(picture_list[3], cmap='gray')
    # axes[1, 2].imshow(picture_list[5], cmap='gray')
    # axes[1, 0].axis('off')
    # axes[1, 1].axis('off')
    # axes[1, 2].axis('off')

    # # 调整子图之间的间距
    # plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # # 显示图片
    # plt.show()








    new_pictures = ground_truth
    print(new_pictures)
    print(picture_list)
    new_pictures.append('saliency/image_1800/000000012993_0.png')

    # 创建3x3的子图布局，并在每个子图位置上绘制对应的图片
    # 从上到下分别是ground truth; sal; nonsal;
    # 从左到右依次是无文本; 非显著文本; 显著文本
    fig, axes = plt.subplots(4, 3)

    axes[0, 0].imshow(plt.imread(new_pictures[3]), cmap='gray')
    # 在第一行展示新增的索引为0、1、2的三张图
    axes[1, 0].imshow(plt.imread(new_pictures[0]), cmap='gray')
    axes[1, 1].imshow(plt.imread(new_pictures[1]), cmap='gray')
    axes[1, 2].imshow(plt.imread(new_pictures[2]), cmap='gray')

    # 将之前的六张图片分别移动到第二行和第三行
    axes[2, 0].imshow(picture_list[0], cmap='gray')
    axes[2, 1].imshow(picture_list[2], cmap='gray')
    axes[2, 2].imshow(picture_list[4], cmap='gray')
    axes[3, 0].imshow(picture_list[1], cmap='gray')
    axes[3, 1].imshow(picture_list[3], cmap='gray')
    axes[3, 2].imshow(picture_list[5], cmap='gray')

    # 隐藏所有子图的坐标轴
    for ax in axes.flatten():
        ax.axis('off')

    fig.text(0.25, 0.9, "Pure", ha='center', va='center', fontsize=12)
    fig.text(0.52, 0.9, "NonSal Text", ha='center', va='center', fontsize=12)
    fig.text(0.77, 0.9, "Sal Text", ha='center', va='center', fontsize=12)
    fig.text(0.52, 0.8, "A man \nin a raincoat \non a bicycle.", ha='center', va='center', fontsize=12, color = 'blue')
    fig.text(0.77, 0.8, "A bus \ndrove over\n a large puddle.", ha='center', va='center', fontsize=12, color = 'green')
    fig.text(0.07, 0.6, "Ground \nTruth", ha='center', va='center', fontsize=10)
    fig.text(0.07, 0.4, "Salient \nGroup", ha='center', va='center', fontsize=10, color = 'red')
    fig.text(0.07, 0.2, "Non-Sal \nGroup", ha='center', va='center', fontsize=10, color = 'orange')
    # 调整子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    # 显示图片
    plt.show()

# print(sal_score_dic)
# print(nonsal_score_dic)

# with open('sal_score_dic.json', 'w') as file:
#     json.dump(sal_score_dic, file)
# with open('nonsal_score_dic.json', 'w') as file:
#     json.dump(nonsal_score_dic, file)

