# 在CV模型本身的预测代码的基础上，加了预测准确率的计算。输出也是一个文件夹里所有图片的预测准确率
import os
import json
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def load_image(img_path, data_transform):
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    return img

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 读取类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 创建模型
    model = AlexNet(num_classes=5).to(device)
    # 加载模型权重
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()

    # 设置图片根路径
    data_root = "../../data_set/flower_data/val_half"
    assert os.path.exists(data_root), "file: '{}' does not exist.".format(data_root)

    total_images = 0
    correct_predictions = 0

    # 遍历每个类别文件夹
    for class_name in os.listdir(data_root):
        class_path = os.path.join(data_root, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_path.endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = load_image(img_path, data_transform)

            with torch.no_grad():
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            total_images += 1
            if class_indict[str(predict_cla)] == class_name:
                correct_predictions += 1

    accuracy = correct_predictions / total_images
    print(f'Total images: {total_images}')
    print(f'Correct predictions: {correct_predictions}')
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
