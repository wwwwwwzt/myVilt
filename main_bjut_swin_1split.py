'''
    -----------------------------BJUT数据集--------------------------------
'''
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
from transformers import AutoImageProcessor,SwinModel,AutoModelForImageClassification
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
import math
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split

# print(dir(transforms))
# exit()
# swin模块来源：https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
swin_processor = AutoImageProcessor.from_pretrained("./weights/swin-tiny-patch4-window7-224")
swin_model = SwinModel.from_pretrained("./weights/swin-tiny-patch4-window7-224")

'''
    -----------------------------数据初始化--------------------------------
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eeg_data_folder = './BJUT/EEGData/'
image_data_folder = "./BJUT/facesFromFrames810/"
# eeg为14个通道
channels = 14
# 3s x 128Hz = 384
samples = 384
# 是否使用扩充数据
use_oversampling = True
'''
    -----------------------------组织EEG数据和label--------------------------------
    1. 得到eeg数据的numpy数组 (trails,samples,channels)  (378, 384, 14)
    2. 得到label的numpy数组 (trails,)  (378,)
    3. 扩充eeg数据  (810, 14, 384)
    4. 扩充后label的数量为270+270+270=810  (810,)
'''
# 存储所有数据的列表和标签列表
data_list = []
label_list = []

# 遍历文件夹中的所有CSV文件
for file_name in os.listdir(eeg_data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(eeg_data_folder, file_name)
        
        # 使用pandas读取CSV文件，只读取2-15列的数据
        df = pd.read_csv(file_path, usecols=list(range(1, 15)), header=0)
        sfreq = 128  # 采样率为128Hz
        ch_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        ch_types = ['eeg'] * channels

        # 创建MNE的Raw对象
        info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
        raw = mne.io.RawArray(df.T, info=info)
        # 滤波
        raw.filter(l_freq=1.0, h_freq=50.0)
        # 创建ICA对象并拟合数据
        ica = ICA(n_components=channels, random_state=0, max_iter=1000)  # 调整参数
        ica.fit(raw)
        # 应用ICA滤波
        ica.exclude = []  # 将排除的独立成分列表设置为空
        ica.apply(raw)
        # 获取滤波后的数据
        data = raw.get_data().T

        # 将数据调整为每个trial的形状
        # 假设数据的样本数为samples 
        num_trials = data.shape[0] // samples
        data = data[:num_trials * samples, :]
        # 重新将数据分成每个trial的形状
        data = data.reshape(num_trials, samples, channels)
        # 添加数据到列表中
        data_list.append(data)
        # 添加标签到列表中
        labels = np.zeros(num_trials, dtype=int)
        labels[3:18] = 1  # Trials 4-18 labeled as 1
        labels[18:] = 2   # Trials 19-21 labeled as 2
        label_list.append(labels)

# 将数据列表转换为numpy数组并按顺序连接
data_array = np.concatenate(data_list, axis=0)
# 将标签列表转换为numpy数组并按顺序连接
label_array = np.concatenate(label_list, axis=0)

print("Data shape before overSampling:", data_array.shape) # (378, 384, 14) 14个通道，每个通道384个样本，总共378个试验数量（18人 x 21个/人）
print("Label shape before overSampling:", label_array.shape) # (378,) 378个试验数量

if use_oversampling:
    # 扩充数据
    ros = RandomOverSampler(sampling_strategy='auto', random_state=30)
    data_array = data_array.reshape(-1, 384 * channels)
    data, labels = ros.fit_resample(data_array, label_array)
else:
    data = data_array
    labels = label_array

data = data.reshape(-1, 384, channels)
data = data.transpose(0, 2, 1)
# print(data.shape) # (810, 14, 384)
# print(labels.shape) # (810,)
# print(labels)
# print(len(labels[labels == 0])) # 270 M 
# print(len(labels[labels == 1])) # 270 N
# print(len(labels[labels == 2])) # 270 A
eeg_data = data
'''
    -----------------------------组织图像数据,与eeg对齐--------------------------------
    1. 得到图像数据的numpy数组 (810, 3, 224, 224)
'''
# 创建一个列表，用于存储所有图像文件的路径
image_file_list = []
for i in range(1, 811):
    filename = f"{image_data_folder}{i}.jpg"
    image_file_list.append(filename)
print("image_file_list:", len(image_file_list)) # 810

# eeg、图片组合在一起
combined_data = list(zip(eeg_data, image_file_list)) # (810, 14, 384) + (810)
# 现在，combined_data是一个长度为810的数组，每个元素都是一个元组，元组中包含一个14x384的数组和一个810的数组
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

# 随机划分训练集和测试集
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=40)

# 分类模型
class MultiModalClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=3, 
                 num_heads=12, dim_feedforward=2048, num_encoder_layers=6,
                 in_c=3, embed_dim=768, patch_size=16, drop_ratio=0.2,
                 eeg_size=384
                 ):
        super(MultiModalClassifier, self).__init__()
        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
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
            batch_size          # 32
            eeg_data            # torch.Size([32, 14, 384]) 
            eeg_embedding       # torch.Size([32, 14, 768])
            image_data          # torch.Size([32, 224, 224, 3])
            image_embedding     # 
            multi_embedding     # 
        '''
        image_data = self.img_processor(images=image_data, return_tensors="pt").to(device)
        
        with torch.no_grad():
            image_embedding = self.swin_model(**image_data)
        image_embedding = image_embedding.last_hidden_state  # torch.Size([32, 49, 768])
    
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

# 创建Dataset
train_dataset = MultiModalDataset(train_data, train_labels)
test_dataset = MultiModalDataset(test_data, test_labels)
# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

epochs = 500
lr = 0.0001
lrf= 0.01
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


start_time = time.time()
writer = SummaryWriter(f'runs/bjut_swin_onceSplit')
for epoch in range(epochs):  # 假设我们训练10个epoch
    model.train()
    for i, (eeg_data, image_data, label) in tqdm(enumerate(train_loader),total=len(train_loader),desc="Training"):
        # 将数据移动到设备上
        eeg_data = eeg_data.to(device)
        image_data = image_data.to(device)
        label = label.to(device)

        # 前向传播
        output = model(eeg_data, image_data)
        # 计算损失
        loss = nn.CrossEntropyLoss()(output, label)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        correct = (predicted == label).sum().item()
        acc = correct / label.size(0)
        writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i) 
        writer.add_scalar('training accuracy', acc, epoch * len(train_loader) + i)  

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}")
    # 在验证集上评估模型
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss = 0
    with torch.no_grad():
        for eeg_data, image_data, label in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            eeg_data, image_data, label = eeg_data.to(device), image_data.to(device), label.to(device)
            outputs = model(eeg_data, image_data)
            _, predicted = torch.max(outputs.data, 1)
            val_total += label.size(0)
            val_correct += (predicted == label).sum().item()

            loss = nn.CrossEntropyLoss()(outputs, label)
            val_loss += loss.item()

        acc = val_correct / val_total
        writer.add_scalar('test accuracy', acc, epoch + 1) # 从0-9变成1-10
        writer.add_scalar('test loss', val_loss / len(test_loader), epoch + 1)
        print("test accuracy", acc)
    # 更新ViT的学习率     
    scheduler.step()

writer.close()

end_time = time.time()
run_time_min = round((end_time - start_time) / 60)
print(f"Total runtime: {run_time_min} minutes")