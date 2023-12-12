import pandas as pd

# 读取Excel文件
df = pd.read_excel('saliency/text最终版本.xlsx', sheet_name='部分-实验设置')

# 获取"image"列的数据
image_column = df['image']

# 获取"描述种类"列的数据
category_column = df['描述种类']

my_dic = dict()

text_column = df['text']

# 遍历每一行
for i in range(len(df)):
    # 获取当前行的"image"和"描述种类"值
    image_value = image_column[i]
    category_value = category_column[i]
    
    # 如果"描述种类"为"整体"
    if category_value == '整体':
        key = str(image_value) + "_0"
        my_dic[key] = ""
        # 创建key，将"image"值与"描述种类"值相加
        key = str(image_value) + "_1"
        
        
    elif category_value == '非显著':
        key = str(image_value) + "_2"
    else:
        key = str(image_value) + "_3"

    my_dic[key] = text_column[i]

    

# 读取Excel文件
df2 = pd.read_excel('saliency/text最终版本.xlsx', sheet_name='整体')

# 获取"image"列的数据
image_column2 = df2['image']



text_column2 = df2['text']

# 遍历每一行
for i in range(len(df2)):
    # 获取当前行的"image"和"描述种类"值
    image_value2 = image_column2[i]
    
    key2 = str(image_value2)
    my_dic[key2] = text_column2[i]

    key2 = str(image_value2) + "_0"
    my_dic[key2] = ""
    





for key in my_dic:
    print(key, end=" ")
    print(my_dic[key])

print(len(my_dic))

import json

# 将字典保存到文件
with open('text.json', 'w') as file:
    json.dump(my_dic, file)



