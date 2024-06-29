import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from model02 import MLP
# 定义设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))


# 定义更深的MLP模型

input_size = 3 * 224 * 224  # Assuming input images are 224x224 with 3 channels
hidden_size = 1024
num_classes = 5

mlp_model = MLP(input_size, hidden_size, num_classes)
mlp_model = mlp_model.to(device)

# 加载模型权重
mlp_model.load_state_dict(torch.load('mlp_model02.pth'))
mlp_model.eval()

# 加载类别映射
with open('../classification/Test2_alexnet/class_indices.json') as f:
    class_indices = json.load(f)

# 图像预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def predict(image_path):
    # 读取图像
    img = Image.open(image_path)
    # 预处理图像
    img = data_transform(img)
    # 添加batch维度
    img = img.unsqueeze(0)
    # 将图像移到设备上
    img = img.to(device)
    # 进行预测
    with torch.no_grad():
        output = mlp_model(img.view(img.size(0), -1))
    # 计算类别概率
    probabilities = torch.softmax(output, dim=1)[0]
    # 根据类别映射获取类别名称和对应概率
    results = []
    for idx, prob in enumerate(probabilities):
        class_name = class_indices[str(idx)]
        results.append({"class": class_name, "prob": prob.item()})
    return results


# 测试预测函数
image_path = 'D:\code_container\pycharm\CV\pics\daisy01.png'
predictions = predict(image_path)
for result in predictions:
    print(f"class: {result['class']}\tprob: {result['prob']:.6f}")
