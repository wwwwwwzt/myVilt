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


# 设置通道数和样本数
channels = 14
samples = 384
# 存放数据的文件夹路径
data_folder = './EEGData/'
# 存储所有数据的列表和标签列表
data_list = []
label_list = []

# 遍历文件夹中的所有CSV文件
for file_name in os.listdir(data_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(data_folder, file_name)
        
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

# 扩充数据
ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
data_array = data_array.reshape(-1, 384*14)
data, labels = ros.fit_resample(data_array, label_array)
# data, labels = data_array, label_array
data = data.reshape(-1,384,14)
data = data.transpose(0,2,1)
print(data.shape) # (810, 14, 384)
print(labels.shape) #(810,)
# print(labels)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(len(labels[labels == 0])) # 270
print(len(labels[labels == 1])) # 270
print(len(labels[labels == 2])) # 270

eeg_data = data

# 创建一个空的numpy数组，用于存放图像数据
image_data = np.empty((810, 3, 224, 224))
# 读取图片并存放到numpy数组中
for i in range(1, 811):
    filename = f"./facesFromFrames_810/{i}.jpg"
    image = Image.open(filename)
    image = image.resize((224, 224))
    # 将图像数据转换为数组
    image_array = np.asarray(image)
    # 将数组添加到image_data中
    image_data[i-1] = image_array.transpose((2, 0, 1))
print("image_data.shape:", image_data.shape) # (810, 3, 224, 224)

# 使用zip函数将它们组合在一起
combined_data = list(zip(eeg_data, image_data)) # (810, 14, 384)+(810, 3, 224, 224)

# 现在，combined_data是一个长度为810的数组，每个元素都是一个元组，元组中包含一个14x384的数组和一个3x224x224的数组
print(len(combined_data))  # 输出：810
print(len(combined_data[0])) # 2
print(combined_data[0][0].shape)  # 输出：(14, 384)
print(combined_data[0][1].shape)  # 输出：(3, 224, 224)

data = combined_data

class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        eeg_data = self.data[index][0].astype(np.float32)
        image_data = self.data[index][1].astype(np.float32)
        label = self.labels[index]
        # print("eeg_data",len(eeg_data)) # 14
        # print("image_data",len(image_data)) # 3
        # print("label",label) # 0\1\2
        return eeg_data, image_data, label

# 随机划分训练集和测试集
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

# 分类模型
class MultiModalClassifier(nn.Module):
    def __init__(self, input_size=768, num_classes=3, 
                 num_heads=16, dim_feedforward=2048, num_encoder_layers=6,
                 in_c=3, embed_dim=768, patch_size=16, drop_ratio=0.2
                 ):
        super(MultiModalClassifier, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size) # [n, 384, 8, 8]
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 13))  # [n, 768, 14, 13]
        
        self.norm = nn.Identity()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size))

        self.fc1 = nn.Linear(input_size, 2048)
        self.fc2 = nn.Linear(2048, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        # 在main768中加入 Upsample，让长度从 3x128=384 增长到768，以便使用vit的权重
        self.projEEG = nn.Upsample(768, mode='linear', align_corners=False)

        self.pos_drop = nn.Dropout(p=drop_ratio)
        # position embedding [1,196+14+1,768]的全0矩阵
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        
    def forward(self, eeg_data, image_data):
        '''
            eeg_data            # torch.Size([30, 14, 384]) 
            eeg_embedding       # torch.Size([30, 14, 768])
            image_data          # torch.Size([30, 3, 224, 224])
            image_embedding     # torch.Size([30, 196, 768])
            multi_embedding     # torch.Size([30, 211, 768]) batch_size,196+14+1,768
        '''
        # image_embedding = self.proj(image_data).flatten(2).transpose(1, 2)
        image_embedding = self.proj(image_data)
        image_embedding = self.adaptive_pool(image_embedding).flatten(2).transpose(1, 2)
        image_embedding = self.norm(image_embedding)

        eeg_embedding = self.projEEG(eeg_data)
        eeg_embedding = self.norm(eeg_embedding) 

        multi_embedding = torch.cat((eeg_embedding, image_embedding), dim=1)

        cls_tokens = self.cls_token.expand(multi_embedding.shape[0], -1, -1)
        multi_embedding = torch.cat((cls_tokens, multi_embedding), dim=1)

        # position embedding
        multi_embedding = self.pos_drop(multi_embedding + self.pos_embed)

        multi_embedding = self.transformer_encoder(multi_embedding)


        # 取出cls token的输出
        multi_embedding = multi_embedding[:, 0, :]
        x = self.fc1(multi_embedding)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.fc2(x)

        return x
    
def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    
classifier = MultiModalClassifier()
classifier = classifier.to(device) 

weights = './weights/vit_base_patch16_224.pth'
if weights != "":
    assert os.path.exists(weights), "weights file: '{}' not exist.".format(weights)
    weights_dict = torch.load(weights, map_location=device)
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] 
    for k in del_keys:
        del weights_dict[k]
    # print(classifier.load_state_dict(weights_dict, strict=False))
    classifier.load_state_dict(weights_dict, strict=False)

# 创建Dataset
train_dataset = MultiModalDataset(train_data, train_labels)
test_dataset = MultiModalDataset(test_data, test_labels)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=30, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=30, shuffle=False)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
total_step = len(train_loader)
for epoch in range(num_epochs):
    correct = 0
    total = 0
    for i, (eeg_data, image_data, labels) in enumerate(train_loader):
        eeg_data = eeg_data.to(device)
        image_data = image_data.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = classifier(eeg_data, image_data)
        loss = criterion(outputs, labels)
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    
    print('Accuracy of the model on the training data: {} %'.format(100 * correct / total))
