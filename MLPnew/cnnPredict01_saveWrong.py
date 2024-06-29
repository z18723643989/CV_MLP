import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
import shutil


# 定义一个更复杂的CNN模型
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(-1, 512 * 14 * 14)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 加载类别索引文件
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)
inverse_class_indices = {v: k for k, v in class_indices.items()}

# 预处理函数
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 预测函数
def predict_image(model, image_path, device):
    image = Image.open(image_path).convert('RGB')
    image = data_transform(image)
    image = image.unsqueeze(0)  # 增加批次维度

    with torch.no_grad():
        output = model(image.to(device))
        _, predicted = torch.max(output, 1)

    return predicted.cpu().numpy()[0]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 加载模型
    model = ComplexCNN().to(device)
    model.load_state_dict(torch.load('cnn_model01.pth'))
    model.eval()

    # 设置数据路径
    data_root = "../data_set/flower_data/val_half"  # 替换为您的数据路径
    output_root = "../data_set/my_model_fault"  # 替换为存放预测错误图片的路径

    # 创建输出目录
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    total_images = 0
    correct_predictions = 0

    for class_name in os.listdir(data_root):
        class_path = os.path.join(data_root, class_name)
        output_class_path = os.path.join(output_root, class_name)

        if not os.path.isdir(class_path):
            continue

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            if not image_path.endswith(('.jpg', '.jpeg', '.png')):
                continue

            total_images += 1
            predicted_class = predict_image(model, image_path, device)
            if predicted_class == int(inverse_class_indices[class_name]):
                correct_predictions += 1
            else:
                # 将错误的图片复制到输出目录
                shutil.copy(image_path, os.path.join(output_class_path, image_name))

    accuracy = correct_predictions / total_images
    print(f"Total images: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    main()
