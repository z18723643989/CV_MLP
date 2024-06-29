import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import vgg

def save_tensors_as_pt(file_path, data):
    torch.save(data, file_path)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # 获取数据根路径
    image_path = os.path.join(data_root, "data_set", "flower_data")  # 花卉数据集路径
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train_half"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将字典写入json文件
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # workers数量
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val_half"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    log_data = []

    for epoch in range(epochs):
        # 训练
        net.train()
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            if step % 10 == 0:  # 记录每10个step的输入输出
                log_data.append({
                    'epoch': epoch + 1,
                    'step': step + 1,
                    'input': images.detach().cpu(),
                    'output': outputs.detach().cpu()
                })

        # 验证
        net.eval()
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))

                if val_step % 10 == 0:  # 记录每10个step的验证输入输出
                    log_data.append({
                        'epoch': epoch + 1,
                        'val_step': val_step + 1,
                        'val_input': val_images.detach().cpu(),
                        'val_output': outputs.detach().cpu()
                    })

        # 保存模型
        acc = sum(torch.eq(torch.max(net(val_images.to(device)), dim=1)[1], val_labels.to(device)).sum().item() for val_images, val_labels in validate_loader) / val_num
        if acc > best_acc:
            best_acc = acc
            torch.save(net.state_dict(), save_path)

    # 写入二进制日志文件
    save_tensors_as_pt('input_output_log_half.pt', log_data)
    print('Finished Training')

if __name__ == '__main__':
    main()