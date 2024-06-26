import scipy.io
import numpy as np
import pandas as pd
import mne
from mne import io
from mne.datasets import sample
from mne.preprocessing import ICA
import os

mat_data = scipy.io.loadmat('./data_preprocessed_matlab/s05.mat') 
original_data = mat_data['data']
original_label = mat_data['labels']

# (40, 40, 8064) 40试验，40通道（前32为eeg），63s（前3秒没用） x 128Hz = 8064
print("original_data形状：", original_data.shape)
# (40, 4) 40试验，4个评价指标
print("original_label形状：", original_label.shape)

'''
    处理eeg，从(40, 40, 8064)变成(40, 32, 7680)
'''
# 前32通道为eeg，去除前3秒
sliced_data = original_data[:, :32, 384:] 
print("sliced_data形状：", sliced_data.shape) # (40, 32, 7680)

eeg_data = sliced_data

'''
    处理labels，从(40, 4)变成(800,)
'''
valence = original_label[:,0]
arousal = original_label[:,1]
# HAHV--1, LAHV--2, LALV--3, HALV--4 完全按照象限来分
VA_labels = np.where((valence > 5) & (arousal > 5), 0,
          np.where((valence >= 5) & (arousal < 5), 1,
            np.where((valence < 5) & (arousal < 5), 2, 3)))
print("V:", valence)
print("A:", arousal)
print(VA_labels)

# 将数据切片成 3 秒一段
segment_size = 3 * 128
# 将 VA_labels 从 40 扩展到 800，VA_labels中的一个数复制20次
num_segments = sliced_data.shape[2] // segment_size # 7680/3/128 = 20
expanded_VA_labels = np.repeat(VA_labels, num_segments)
print(expanded_VA_labels.shape) # (800,)
labels = expanded_VA_labels 
# print(labels)


'''
    处理eeg数据
'''
sfreq = 128  # 采样率为128Hz
channels = 32
samples = 384
num_each_trial = 20
ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7', 'P3', 
            'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2', 'C4', 'T8', 
            'FC6', 'FC2', 'F4', 'F8', 'AF4', 'FP2', 'Fz', 'Cz']
ch_types = ['eeg'] * channels

data_list = []
eeg_data_segments = np.split(eeg_data, 40, axis=0) # (40, 32, 7680)
index = 0
for segment in eeg_data_segments:
    index = index + 1
    print("wzt index", index)
    segment_2d = segment.reshape(-1, channels).T
    print("wzt segment_2d", segment_2d.shape) # (32, 7680)
    # 创建MNE的Raw对象
    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(segment_2d, info=info)
    # 滤波
    raw.filter(l_freq=1.0, h_freq=50.0)
    # 创建ICA对象并拟合数据
    ica = ICA(n_components=channels, random_state=0, max_iter=1000)  # 调整参数
    ica.fit(raw)
    # 应用ICA滤波
    ica.exclude = []  # 将排除的独立成分列表设置为空
    ica.apply(raw)
    # 获取滤波后的数据
    data = raw.get_data().T # (7680, 32)

    # 将数据调整为每个trial的形状
    data = data[:num_each_trial * samples, :]
    # 重新将数据分成每个trial的形状
    data = data.reshape(num_each_trial, samples, channels)
    # 添加数据到列表中
    data_list.append(data)

# 将数据列表转换为numpy数组并按顺序连接
data_array = np.concatenate(data_list, axis=0) # (800, 384, 32)
# 交换位置
data_array = np.swapaxes(data_array, 1, 2) # (800, 32, 384)
print(data_array.shape) 
eeg_data = data_array

'''
    保存数据
'''
np.save('./EEGData/s05_eeg.npy', eeg_data) # (800, 32, 384)
np.save('./EEGData/s05_labels.npy', labels) # (800,)