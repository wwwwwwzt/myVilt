import os
import pandas as pd
import numpy as np
import numpy as np

# mne imports
import mne
from mne import io
from mne.datasets import sample
from mne.preprocessing import ICA


# tools for plotting confusion matrices
from matplotlib import pyplot as plt

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

# 打印数据的形状
print("Data shape:", data_array.shape)
print("Label shape:", label_array.shape)

from itertools import combinations
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

data = np.array(data_array)
labels = np.array(label_array)
print(data.shape)
print(labels.shape)

print(len(labels[labels == 0]))
print(len(labels[labels == 1]))
print(len(labels[labels == 2]))

ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
# data = data.reshape(-1, 384*14)
# data, labels = ros.fit_resample(data, labels)


# 随机划分训练集和测试集
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.3, random_state=42)

class TransformerModel1(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads=14, dim_feedforward=2048, num_encoder_layers=6):
        super(TransformerModel1, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            # 384 heads dim
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        # 14+9
        self.fc = nn.Linear(input_size*11, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
# 分类模型
class EEGClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(EEGClassifier, self).__init__()
        # 加载对比学习模型
        contrastive_model = temporal_channel_joint_attention()
        contrastive_model.load_state_dict(torch.load('./contrastive_learning_model.pth'))

        # 冻结对比学习模型的encoder_temproal和encoder_channel部分的参数
        for param in contrastive_model.encoder_temproal.parameters():
            param.requires_grad = True
        for param in contrastive_model.encoder_channel.parameters():
            param.requires_grad = True

        # 提取encoder_temproal和encoder_channel
        self.encoder_temproal = contrastive_model.encoder_temproal
        # self.encoder_channel = contrastive_model.encoder_channel

        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()
        self.BN = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(num_windows*num_channels*num_FBs, 2048)
        self.fc2 = nn.Linear(2048, 504)
        self.fc3 = nn.Linear(504, num_classes)


    def forward(self, PSDs):
        temporal_tokens = PSDs.reshape(-1, num_windows, num_channels*num_FBs)
        # channel_tokens = PSDs.reshape(-1, num_channels, num_windows*num_FBs)
        temporal_features = self.encoder_temproal(temporal_tokens)
        # channel_features = self.encoder_channel(channel_tokens)
        # # concat 
        # temporal_features = temporal_features.reshape(-1, 1, num_windows*num_channels*num_FBs)
        # channel_features = channel_features.reshape(-1, 1, num_channels*num_windows*num_FBs)
        # joint_features = torch.cat((temporal_features, channel_features), dim=1)
        # # classification
        temporal_features = nn.Flatten()(temporal_features)
        print(temporal_features.shape)
        joint_features = self.fc1(temporal_features)
        joint_features = self.BN(joint_features)
        joint_features = self.activation(joint_features)
        joint_features = self.dropout(joint_features)
        joint_features = self.fc2(joint_features)
        joint_features = self.activation(joint_features)
        joint_features = self.dropout(joint_features)
        res = self.fc3(joint_features)
        # joint_features = self.activation(joint_features)
        # res = self.dropout(joint_features)
        return res
    
# 创建分类模型
classifier = EEGClassifier().to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(classifier.parameters())
criterion = nn.CrossEntropyLoss()

# 定义训练函数
def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.clone().detach().type(torch.float32)
        labels = labels.clone().detach().type(torch.long)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_acc.double() / len(train_loader.dataset)
    return train_loss, train_acc

# 定义测试函数
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    y_true = []
    y_pred = []
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.clone().detach().type(torch.float32)
        labels = labels.clone().detach().type(torch.long)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        test_acc += torch.sum(preds == labels.data)
        y_true.extend(labels.data.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_acc.double() / len(test_loader.dataset)
    print(classification_report(y_true, y_pred))
    return test_loss, test_acc

# 定义EEG数据集类
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        eeg = self.data[index]
        label = self.labels[index]
        return eeg, label

# 创建Dataset
train_dataset = EEGDataset(train_data, train_labels)
test_dataset = EEGDataset(test_data, test_labels)

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(classifier, train_loader, optimizer, criterion)
    test_loss, test_acc = test(classifier, test_loader, criterion)
    print("Epoch: {} Train Loss: {:.6f} Train Acc: {:.6f} Test Loss: {:.6f} Test Acc: {:.6f}".format(epoch, train_loss, train_acc, test_loss, test_acc))
    
print("Training finished!")

# 保存模型
torch.save(classifier.state_dict(), './simCLR_classifier.pth')
