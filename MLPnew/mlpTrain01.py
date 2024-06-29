import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model01 import MLP

# 检查 GPU 是否可用
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU")

print("Using device:", device)
if device.type == 'cuda':
    print("Device name:", torch.cuda.get_device_name(device))

# 加载数据
train_inputs = np.load('data_set/train_inputs.npy')
train_outputs = np.load('data_set/train_outputs.npy')
val_inputs = np.load('data_set/val_inputs.npy')
val_outputs = np.load('data_set/val_outputs.npy')

# 将数据转换为Tensor
train_inputs = torch.tensor(train_inputs, dtype=torch.float32)
train_outputs = torch.tensor(train_outputs, dtype=torch.float32)
val_inputs = torch.tensor(val_inputs, dtype=torch.float32)
val_outputs = torch.tensor(val_outputs, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_outputs)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



input_size = train_inputs.shape[1] * train_inputs.shape[2] * train_inputs.shape[3]
hidden_size = 512
num_classes = train_outputs.shape[1]

mlp_model = MLP(input_size, hidden_size, num_classes)
mlp_model = mlp_model.to(device)

# 损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# 训练MLP模型
epochs = 10
for epoch in range(epochs):
    mlp_model.train()
    running_loss = 0.0
    for inputs, outputs in train_loader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        outputs = outputs.to(device)

        optimizer.zero_grad()
        predictions = mlp_model(inputs)
        loss = criterion(predictions, outputs)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}')

# 验证MLP模型
mlp_model.eval()
val_loss = 0.0
with torch.no_grad():
    for inputs, outputs in val_loader:
        inputs = inputs.view(inputs.size(0), -1).to(device)
        outputs = outputs.to(device)

        predictions = mlp_model(inputs)
        loss = criterion(predictions, outputs)

        val_loss += loss.item()

print(f'Validation Loss: {val_loss / len(val_loader):.4f}')

# 保存MLP模型
torch.save(mlp_model.state_dict(), 'mlp_model01.pth')
print('MLP Model Saved')
