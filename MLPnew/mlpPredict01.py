import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model01 import MLP  # 假设 MLP 定义在 model01.py 中

def main():
    # 定义设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 图像预处理
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载类别映射
    json_path = '../classification/Test2_alexnet/class_indices.json'
    assert os.path.exists(json_path), "file: '{}' does not exist.".format(json_path)
    with open(json_path, 'r') as f:
        class_indices = json.load(f)

    # 定义MLP模型
    input_size = 3 * 224 * 224  # Assuming input images are 224x224 with 3 channels
    hidden_size = 512
    num_classes = 5

    mlp_model = MLP(input_size, hidden_size, num_classes)
    mlp_model = mlp_model.to(device)

    # 加载模型权重
    weights_path = 'mlp_model01.pth'
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    mlp_model.load_state_dict(torch.load(weights_path))
    mlp_model.eval()

    # 加载图像
    img_path = '../pics/daisy01.png'  # 修改为你想预测的图像路径
    assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output = mlp_model(img.view(img.size(0), -1).to(device))
        probabilities = torch.softmax(output, dim=1)[0]

    predict_cla = torch.argmax(probabilities).item()
    print_res = "class: {}   prob: {:.3f}".format(class_indices[str(predict_cla)], probabilities[predict_cla].item())
    plt.title(print_res)

    # 设定概率阈值，避免显示极小概率
    threshold = 0.01
    for idx, prob in enumerate(probabilities):
        if prob.item() > threshold:
            print("class: {:10}   prob: {:.3f}".format(class_indices[str(idx)], prob.item()))

    plt.show()

if __name__ == '__main__':
    main()
