import math
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel, SwinModel
from tqdm import tqdm
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch.optim.lr_scheduler as lr_scheduler
import random
'''
    -----------------------------超参数定义--------------------------------
'''
# swin模块来源：https://huggingface.co/microsoft/swin-tiny-patch4-window7-224
swin_processor = AutoImageProcessor.from_pretrained("./weights/swin-tiny-patch4-window7-224")
swin_model = SwinModel.from_pretrained("./weights/swin-tiny-patch4-window7-224")

# swin_finetuned模块来源：https://huggingface.co/MahmoudWSegni/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12_right
# swin_processor = AutoImageProcessor.from_pretrained("./weights/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12")
# swin_model = SwinModel.from_pretrained("./weights/swin-tiny-patch4-window7-224-finetuned-face-emotion-v12")

test_id = '1'
tb_dir = "runs/deap_swin_dep"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
channels = 32
samples = 384
eeg_data_folder = './DEAP/EEGData/'
test_eeg_data = np.load(f"{eeg_data_folder}s{str(test_id).zfill(2)}_eeg.npy")
test_labels = np.load(f"{eeg_data_folder}s{str(test_id).zfill(2)}_labels.npy")
available_subjects = [1, 2, 6, 7, 8, 9, 10, 12, 13, 16, 17, 18, 19, 20, 21, 22]
train_ids = [subject for subject in available_subjects if str(subject) != test_id]
'''
    -----------------------------合并训练集的所有eeg和label--------------------------------
'''
train_eeg_data = []
train_labels = []
for i in train_ids:
    eeg_file = f"{eeg_data_folder}s{str(i).zfill(2)}_eeg.npy"
    label_file = f"{eeg_data_folder}s{str(i).zfill(2)}_labels.npy"
    # 尝试加载 EEG 数据和标签，如果文件不存在则跳过
    try:
        eeg_data = np.load(eeg_file)
        labels = np.load(label_file)
        
        train_eeg_data.append(eeg_data)
        train_labels.append(labels)
    except FileNotFoundError:
        print(f"Data for subject {i} not found!!!!!!!!!!!!!!!!!")

train_eeg_data = np.concatenate(train_eeg_data, axis=0)
train_labels = np.concatenate(train_labels, axis=0)
'''
    -----------------------------组织图像数据,与eeg对齐--------------------------------
'''
# 训练集
train_image_file_list = []
for i in train_ids:
    person_image_data_folder = f"./DEAP/faces/s{str(i).zfill(2)}/" 
    # 遍历每个人的图像文件
    for j in range(1, 801):
        filename = f"{person_image_data_folder}{j}.jpg"
        train_image_file_list.append(filename)

train_combined_data = list(zip(train_eeg_data, train_image_file_list))

# 测试集
test_image_file_list = []
person_image_data_folder = f"./DEAP/faces/s{str(test_id).zfill(2)}/" 
for j in range(1, 801):
    filename = f"{person_image_data_folder}{j}.jpg"
    test_image_file_list.append(filename)
test_combined_data = list(zip(test_eeg_data, test_image_file_list))

# data = train_combined_data
train_data = train_combined_data
train_labels = train_labels
test_data = test_combined_data
test_labels = test_labels

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(0)

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
                 num_heads=12, dim_feedforward=2048, num_encoder_layers=6, device=device, 
                 eeg_size=384, transformer_dropout_rate=0.2, cls_dropout_rate=0.2
                 ):
        super(MultiModalClassifier, self).__init__()
        self.transformer_dropout_rate = transformer_dropout_rate
        self.cls_dropout_rate = cls_dropout_rate
        self.img_processor = swin_processor
        self.swin_model = swin_model
        for param in self.swin_model.parameters():
            param.requires_grad = True

        self.token_type_embeddings = nn.Embedding(2, input_size)
        
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=transformer_dropout_rate, batch_first=True),
            num_layers=num_encoder_layers
        )

        self.eeg_proj = nn.Linear(eeg_size, input_size)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(eeg_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_size)).to(device)
        self.dropout = nn.Dropout(cls_dropout_rate)
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
        image_embedding = self.swin_model(**image_data)
        image_embedding = image_embedding.last_hidden_state

        eeg_data = self.layernorm(eeg_data)
        eeg_embedding = self.eeg_proj(eeg_data)
        eeg_embedding = self.activation(eeg_embedding)

        image_embedding, eeg_embedding = (
            image_embedding + self.token_type_embeddings(torch.zeros(image_embedding.shape[0], 1, dtype=torch.long, device=device)),
            eeg_embedding + self.token_type_embeddings(torch.ones(eeg_embedding.shape[0], 1, dtype=torch.long, device=device))
        )

        multi_embedding = torch.cat((image_embedding, eeg_embedding), dim=1)

        multi_embedding = torch.cat((self.cls_token.expand(multi_embedding.size(0), -1, -1), multi_embedding), dim=1)

        multi_embedding = self.transformer_encoder(multi_embedding)

        # 取出cls token的输出
        cls_token_output = image_embedding[:, 0, :]
        cls_token_output = self.dropout(cls_token_output)

        x = self.classifier(cls_token_output)

        return x

model = MultiModalClassifier().to(device) 

epochs = 100
lr = 0.0001
lrf= 0.01
max_lr = 0.00001

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# ViT学习率
lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# 计算并打印每个类别的数量
unique_train, counts_train = np.unique(train_labels, return_counts=True)
unique_test, counts_test = np.unique(test_labels, return_counts=True)
print("训练集中每个类别的数量：", dict(zip(unique_train, counts_train)))
print("测试集中每个类别的数量：", dict(zip(unique_test, counts_test)))

# 创建数据集、数据加载器
train_dataset = MultiModalDataset(data=train_data, labels=train_labels)
test_dataset = MultiModalDataset(data=test_data, labels=test_labels)
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader.dataset) // train_loader.batch_size, epochs=epochs, pct_start=0.20)
# print("steps_per_epoch:", len(train_loader.dataset) // train_loader.batch_size)

start_time = datetime.now()
writer = SummaryWriter(f'{tb_dir}')
writer.add_scalar('transformer dropout', model.transformer_dropout_rate, global_step=0)
writer.add_scalar('cls dropout', model.cls_dropout_rate, global_step=0)
test_accuracies = []
for epoch in range(epochs):
    train_bar = tqdm(enumerate(train_loader),total=len(train_loader),desc="Training", leave=False)
    model.train()

    for i, (eeg_data, image_data, label) in enumerate(train_loader):
        eeg_data = eeg_data.to(device)
        image_data = image_data.to(device)
        label = label.to(device)

        output = model(eeg_data, image_data)
        loss = nn.CrossEntropyLoss()(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新warm-up学习率 
        # scheduler.step()

        if i % 10 == 0:  # 每10个批次，记录损失和准确率
            _, predicted = torch.max(output, 1)
            correct = (predicted == label).sum().item()
            acc = correct / label.size(0)
            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + i)  # loss 800
            writer.add_scalar('training accuracy', acc, epoch * len(train_loader) + i)  # acc 800

        train_bar.update(1)
        train_bar.write(f"Epoch: {epoch + 1}, Training Loss: {loss.item()}")

    # 在测试集上评估模型
    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0
    with torch.no_grad():
        for i, (eeg_data, image_data, label) in enumerate(test_loader):
            eeg_data, image_data, label = eeg_data.to(device), image_data.to(device), label.to(device)
            outputs = model(eeg_data, image_data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += label.size(0)
            test_correct += (predicted == label).sum().item()

            # 计算测试集的损失
            loss = nn.CrossEntropyLoss()(outputs, label)
            test_loss += loss.item()

        acc = test_correct / test_total
        test_accuracies.append(acc)
        writer.add_scalar('test accuracy', acc, epoch + 1) # 从0-9变成1-10
        writer.add_scalar('test loss', test_loss / len(test_loader), epoch + 1)
        print(f"Accuracy: {acc}")
    # 更新ViT的学习率 
    writer.add_scalar('learning rate', optimizer.param_groups[0]["lr"], epoch)
    scheduler.step()

writer.close()

print("\n--------------------------训练结束--------------------------")
print("Max accuracy:", max(test_accuracies))

end_time = datetime.now()
run_time = end_time - start_time
run_time_seconds = run_time.total_seconds()  # 获取总秒数
run_time_min = round(run_time_seconds / 60)  # 转换为分钟
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {run_time_min} minutes")

with open('./第六轮/第六轮记录.md', 'a') as f:
    f.write('{}  {}\n'.format(id, max(test_accuracies)))