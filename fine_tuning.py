# import json
# import os
# import torch
# import torchvision
# from torch import nn
# from d2l import torch as d2l
# from torchvision.models import ResNet18_Weights
#
# # @save
# d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
#                           'fba480ffa8aa7e0febbb511d181409f899b9baa5')
#
# data_dir = d2l.download_extract('hotdog')
# train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
# test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))
#
# hotdogs = [train_imgs[i][0] for i in range(8)]
# not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)
#
# # 创建一个用于对训练图像进行数据增强和标准化的数据转换序列
# normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
#                                              [0.229, 0.224, 0.225])  # 第一个是rgb通道分别的均值，第二个是rgb通道的标准差
# train_augs = torchvision.transforms.Compose([
#     torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪和缩放图像到 224x224 尺寸
#     torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
#     torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
#     normalize  # 对图像进行标准化处理
# ])
#
# # 创建一个用于对测试图像进行标准化的数据转换序列
# test_augs = torchvision.transforms.Compose([
#     torchvision.transforms.Resize([256, 256]),  # 将图像调整为 256x256 尺寸
#     torchvision.transforms.CenterCrop(224),  # 从中心裁剪出 224x224 大小的图像
#     torchvision.transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
#     normalize  # 对图像进行标准化处理，与训练数据使用相同的均值和标准差
# ])
#
# pretrained_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
# # var = pretrained_net.fc
# # print(var)
#
# finetune_net = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
# finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
# nn.init.xavier_uniform_(
#     finetune_net.fc.weight)  # 这一行代码用 Xavier 初始化方法初始化了新添加的全连接层的权重。Xavier初始化是一种用于初始化神经网络权重的常用方法，有助于确保模型的初始权重适合训练。
#
#
# def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
#     # 创建训练数据迭代器
#     train_iter = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(
#         os.path.join(data_dir, 'train'), transform=train_augs),
#         batch_size=batch_size, shuffle=True)
#
#     # 创建测试数据迭代器
#     test_iter = torch.utils.data.DataLoader(
#         torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs), batch_size=batch_size)
#
#     # 获取所有可用的GPU设备（如果有的话）
#     devices = d2l.try_all_gpus()
#
#     # 定义损失函数为交叉熵损失
#     loss = nn.CrossEntropyLoss(reduction="none")
#
#     # 根据参数组设置选择不同的优化器和学习率策略
#     if param_group:
#         # 如果 param_group 为 True，则将网络参数分成两个组进行优化
#         # 一个组是除了全连接层之外的参数，另一个组是全连接层的参数
#         params_1x = [param for name, param in net.named_parameters()
#                      if name not in ["fc.weight", "fc.bias"]]
#         trainer = torch.optim.SGD([{'params': params_1x}, {'params': net.fc.parameters(), 'lr': learning_rate * 10}],
#                                   lr=learning_rate, weight_decay=0.001)
#     else:
#         # 如果 param_group 为 False，则优化所有网络参数
#         trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
#
#     # 调用训练函数，开始模型微调
#     d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
#
#
# train_fine_tuning(finetune_net, 5e-5)
import json
s = {
    "url":"https://wdtrip-videos.oss-cn-hangzhou.aliyuncs.com/2023/07/04/a3b22c47-146e-4b85-881a-9e1aeaa41ea1.jpg",
    "type":"pic"
}

print(json.dumps(s))