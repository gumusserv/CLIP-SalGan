import json
import os

# 用于获取每个图片-文本对的图片路径, 文本描述
def get_Data(image_directory_path, target_directory_path):

    # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
    image_paths = [image_directory_path + "/" + f for f in os.listdir(image_directory_path) if os.path.isfile(os.path.join(image_directory_path, f))]

    image_paths = image_paths[0 : ]

    # image_paths = [item for item in image_paths if "_1.png" in item  or "_0.png" in item or ("_2" not in item  and "_3" not in item)]
    # image_paths = [item for item in image_paths if "_2.png" in item  or "_0.png" in item]
    # new_image_paths = []
    
    # for string in image_paths:
    #     if "_0" in string:
    #         # 在包含"_0"的字符串中查找并替换为"_3"
    #         updated_string = string.replace("_0", "_2")
            
    #         if updated_string in image_paths:
    #             # 删除最初包含"_0"的字符串
    #             new_image_paths.append(string)
    #     else:
    #         new_image_paths.append(string)

    # image_paths = new_image_paths
            
    # 如果你只想获取文件而不包括子目录，可以使用下面的代码来过滤
    target_paths = [target_directory_path + "/" + f for f in os.listdir(target_directory_path) if os.path.isfile(os.path.join(target_directory_path, f))]

    target_paths = target_paths[0 : ]

    # target_paths = [item for item in target_paths if "_1.png" in item  or "_0.png" in item or ("_2" not in item  and "_3" not in item)]
    # target_paths = [item for item in target_paths if "_2.png" in item  or "_0.png" in item]
    # new_target_paths = []
    
    # for string in target_paths:
    #     if "_0" in string:
    #         # 在包含"_0"的字符串中查找并替换为"_3"
    #         updated_string = string.replace("_0", "_2")
            
    #         if updated_string in target_paths:
    #             # 删除最初包含"_0"的字符串
    #             new_target_paths.append(string)
    #     else:
    #         new_target_paths.append(string)

    # target_paths = new_target_paths

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
    # print(image_paths)
    # print(target_paths)
    # print(text_descriptions)
    # print(len(image_paths))
    # for _ in image_paths:
    #     print(_)
    # index_list = []
    # for i in range(1800):
        # if "_0.png" not in image_paths[i] and "_1.png" not in image_paths[i] and "_2.png" not in image_paths[i] and "_3.png" not in image_paths[i]:
        # if "_3.png" in image_paths[i]:
            # index_list.append(i - 3)
            # index_list.append(i - 2)
            # index_list.append(i - 1)
            # index_list.append(i)
    # for j in index_list:
    #     print(image_paths[j])
    # print(len(index_list))
    # print(index_list)

    # for i in range(0, 8):
    #     print(image_paths[i])
    #     print(target_paths[i])
    #     print(text_descriptions[i])
    #     print(image_paths[i][image_paths[i].rfind('/') :] == target_paths[i][target_paths[i].rfind('/') :])
    return image_paths, target_paths, text_descriptions

# image_directory_path = 'saliency/image_1800'
# target_directory_path = 'saliency/map_1800'

# image_paths, target_paths, text_descriptions = get_Data(image_directory_path, target_directory_path)
