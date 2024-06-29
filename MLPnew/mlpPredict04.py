import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json


# 定义一个更复杂的MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# 加载类别映射文件
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load('mlp_model04.pth'))
model.eval()

# 定义数据预处理
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
