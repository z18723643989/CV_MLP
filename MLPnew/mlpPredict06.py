import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


# 定义一个更复杂的MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(224 * 224 * 3, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, 5)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout(x)
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x


# 加载训练好的模型权重
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load('mlp_model06.pth'))
model.eval()

# 类别映射
label_map = {
    "daisy": 0,
    "dandelion": 1,
    "roses": 2,
    "sunflowers": 3,
    "tulips": 4
}

# 反向映射
reverse_label_map = {v: k for k, v in label_map.items()}

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# 预测单张图片的类型
def predict(image_path, model, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()


# 批量预测并计算成功率
def evaluate_folder(root_folder, model, transform, device, label_map):
    total_images = 0
    correct_predictions = 0

    for label_name, label_index in label_map.items():
        folder_path = os.path.join(root_folder, label_name)
        if not os.path.exists(folder_path):
            continue
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if not os.path.isfile(image_path):
                continue
            predicted_label = predict(image_path, model, transform, device)
            total_images += 1
            if predicted_label == label_index:
                correct_predictions += 1

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    return accuracy, correct_predictions, total_images



# 示例：预测文件夹中的图片类别并计算成功率
root_folder = '../data_set/flower_data/val_half'
accuracy, correct_predictions, total_images = evaluate_folder(root_folder, model, transform, device, label_map)
print(f'Total images: {total_images}')
print(f'Correct predictions: {correct_predictions}')
print(f'Accuracy: {accuracy:.2f}')
