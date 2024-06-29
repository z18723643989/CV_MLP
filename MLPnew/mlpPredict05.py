# predict.py
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json


# 使用预训练的ResNet18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# 加载类别映射文件
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=5).to(device)
model.load_state_dict(torch.load('mlp_model05.pth'))
model.eval()

# 定义数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image)
    image = image.unsqueeze(0)  # 添加批次维度

    with torch.no_grad():
        outputs = model(image.to(device))
        _, predicted = torch.max(outputs, 1)
        class_id = predicted.item()
        class_name = class_indices[str(class_id)]
        return class_name


# 遍历文件夹，预测所有图片并计算准确率
def evaluate_model(dataset_path):
    total_images = 0
    correct_predictions = 0

    for class_name in os.listdir(dataset_path):
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if not os.path.isfile(image_path):
                continue

            predicted_class = predict_image(image_path)
            if predicted_class == class_name:
                correct_predictions += 1
            total_images += 1

    accuracy = correct_predictions / total_images
    print(f'Total images: {total_images}')
    print(f'Correct predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.2f}')


# 示例：预测整个数据集
dataset_path = '../data_set/flower_data/val_half'  # 替换为您的数据集路径
evaluate_model(dataset_path)
