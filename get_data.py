import json
import os

# 用于获取每个图片-文本对的图片路径, 文本描述
def get_Data(image_directory_path, target_directory_path):

    # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
    image_paths = [image_directory_path + "/" + f for f in os.listdir(image_directory_path) if os.path.isfile(os.path.join(image_directory_path, f))]

    image_paths = image_paths[0 : ]

    # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
    target_paths = [target_directory_path + "/" + f for f in os.listdir(target_directory_path) if os.path.isfile(os.path.join(target_directory_path, f))]

    target_paths = target_paths[0 : ]


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
