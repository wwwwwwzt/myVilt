import os
import pandas as pd
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from mne.preprocessing import ICA
from matplotlib import pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm
from torchvision import transforms
from sklearn.model_selection import KFold

'''
    -----------------------------数据初始化--------------------------------
'''
# vit模块来源：https://huggingface.co/google/vit-base-patch16-224/tree/main
processor = AutoImageProcessor.from_pretrained("/home/zcl/wzt/try/weights/vit-base-patch16-224")
vitmodel = AutoModel.from_pretrained("/home/zcl/wzt/try/weights/vit-base-patch16-224")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eeg_data_folder = './DEAP/EEGData/'
image_data_folder = "./DEAP/faces/"
channels = 32
# 3s x 128Hz = 384
samples = 384
eeg_data = np.load(f"{eeg_data_folder}s01_eeg.npy")
labels = np.load(f"{eeg_data_folder}s01_labels.npy")
image_data = np.empty((800, 3, 224, 224))

'''
    -----------------------------组织图像数据,与eeg对齐--------------------------------
    1. 得到图像数据的numpy数组 (800, 3, 224, 224)
'''
# 创建一个列表，用于存储所有图像文件的路径
image_file_list = []
for i in range(1, 801):
    filename = f"{image_data_folder}{i}.jpg"
    image_file_list.append(filename)
print("image_file_list:", len(image_file_list)) # 800

# eeg、图片组合在一起
combined_data = list(zip(eeg_data, image_file_list)) # (800, 14, 384) + (800)
# 现在，combined_data是一个长度为800的数组，每个元素都是一个元组，元组中包含一个14x384的数组和一个800的数组
data = combined_data

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        eeg_data = self.data[index][0].astype(np.float32)
        image_file = self.data[index][1]
        image_data = np.array(Image.open(image_file)).astype(np.float32)
        image_data = transforms.Resize((224,224))(Image.open(image_file))
        image_data = np.array(image_data).astype(np.float32)
        label = self.labels[index]
        return eeg_data, image_data, label

# 分类模型
class MultiModalClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=4, 
                 num_heads=12, dim_feedforward=2048, num_encoder_layers=6, 
                 device=device, eeg_size=384
                 ):
        super(MultiModalClassifier, self).__init__()
        self.img_processor = processor
        self.vit_model = vitmodel
        for param in self.vit_model.parameters():
            param.requires_grad = True

        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.eeg_proj = nn.Linear(eeg_size, input_size)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(eeg_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(device)

        self.classifier = nn.Linear(input_size,num_classes)

        
    def forward(self, eeg_data, image_data):
        '''
            eeg_data            # torch.Size([30, 14, 384]) 
            eeg_embedding       # torch.Size([30, 14, 768])
            image_data          # torch.Size([30, 3, 224, 224])
            image_embedding     # torch.Size([30, 196, 768])
            multi_embedding     # torch.Size([30, 211, 768]) batch_size,196+14+1,768
        '''
        image_data = self.img_processor(image_data, return_tensors="pt").to(device)
        image_embedding = self.vit_model(**image_data).last_hidden_state[:,1:,:]

        eeg_data = self.layernorm(eeg_data)
        eeg_embedding = self.eeg_proj(eeg_data)
        eeg_embedding = self.activation(eeg_embedding)

        image_embedding, eeg_embedding = (
            image_embedding + self.token_type_embeddings(torch.zeros(image_embedding.shape[0], 1, dtype=torch.long, device=device)),
            eeg_embedding + self.token_type_embeddings(torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=device))
        )

        multi_embedding = torch.cat((image_embedding, eeg_embedding), dim=1)

        multi_embedding = torch.cat((self.cls_token.expand(multi_embedding.size(0), -1, -1), multi_embedding), dim=1)

        multi_embedding = self.transformer_encoder(multi_embedding)  # torch.Size([30, 211, 768])

        # 取出cls token的输出
        multi_embedding = image_embedding[:, 0, :]
        x = self.classifier(multi_embedding)

        return x
    

model = MultiModalClassifier().to(device) 

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# 创建一个KFold对象
kf = KFold(n_splits=5, shuffle=True, random_state=30)

epochs = 10
# 用于存储每次迭代的准确率
accuracies = []

# 开始交叉验证
for fold, (train_index, test_index) in tqdm(enumerate(kf.split(data)), total=5, desc="Cross Validation"):
    print(f"Fold {fold + 1}")

    # 分割数据
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    train_labels = labels[train_index]

    print(max(test_index)) # 798
    print(len(labels))  # 800
    assert max(test_index) < len(labels), "Index out of range"
    test_labels = labels[test_index]
    # test_labels = labels[test_index]

    # 创建数据集
    train_dataset = MultiModalDataset(train_data, train_labels)
    test_dataset = MultiModalDataset(test_data, test_labels)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 训练和验证模型
    for epoch in range(epochs):  # 假设我们训练10个epoch
        # 训练阶段
        model.train()
        for eeg_data, image_data, label in tqdm(train_loader,total=len(train_loader),desc="Training"):
            # 将数据移动到设备上
            eeg_data = eeg_data.to(device)
            image_data = image_data.to(device)
            label = label.to(device)

            # 前向传播
            output = model(eeg_data, image_data)

            # print("output:", output.shape) # torch.Size([32, 4]) 
            # print("output:", label.shape) # torch.Size([32])

            # 计算损失
            loss = nn.CrossEntropyLoss()(output, label)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")

    # 在验证集上评估模型
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for eeg_data, image_data, label in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            eeg_data, image_data, label = eeg_data.to(device), image_data.to(device), label.to(device)
            # eeg_data, image_data, labels = torch.from_numpy(eeg_data).to(device), torch.from_numpy(image_data).to(device), torch.from_numpy(label).to(device)
            outputs = model(eeg_data, image_data)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    # 计算准确率并添加到列表中
    accuracy = correct / total
    print(f"Accuracy: {accuracy}")
    accuracies.append(accuracy)

# 打印所有的准确率以及最大的准确率
print('Accuracies:', accuracies)
print('Max accuracy:', max(accuracies))