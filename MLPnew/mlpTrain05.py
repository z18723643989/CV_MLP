# 在mlpTrain04的基础上，使用预训练的ResNet18模型
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms
import json

# 加载保存的数据
train_inputs = np.load('data_set/train_inputs.npy')
train_outputs = np.load('data_set/train_outputs.npy')
val_inputs = np.load('data_set/val_inputs.npy')
val_outputs = np.load('data_set/val_outputs.npy')

# 转换为PyTorch张量
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_outputs = torch.tensor(np.argmax(train_outputs, axis=1), dtype=torch.long)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_outputs = torch.tensor(np.argmax(val_outputs, axis=1), dtype=torch.long)

# 创建数据加载器
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_outputs)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class ResNet18(nn.Module):
    def __init__(self, num_classes=5):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数和优化器
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练和验证过程
epochs = 20  # 增加训练轮次
best_val_acc = 0.0
best_model_path = 'mlp_model05.pth'

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()  # 调整学习率

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {val_acc:.2f}')

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

print('Finished Training')
