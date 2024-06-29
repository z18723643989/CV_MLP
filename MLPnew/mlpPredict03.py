import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# 定义数据变换
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载测试数据集
test_data_path = '../data_set/flower_data/val_half'  # 替换为您的测试数据路径
test_dataset = datasets.ImageFolder(root=test_data_path, transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 加载模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load('mlp_model03.pth'))
model.eval()

# 预测
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 计算总图片数和正确预测数
total_images = len(all_labels)
correct_predictions = sum(p == l for p, l in zip(all_preds, all_labels))

# 计算成功率
accuracy = accuracy_score(all_labels, all_preds)
print(f'Total images: {total_images}')
print(f'Correct predictions: {correct_predictions}')
print(f'Test Accuracy: {accuracy:.4f}')

