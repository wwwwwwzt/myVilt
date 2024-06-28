# myVilt

这是一个多模态学习仓库，基于 Vilt/ViT/Swin-Transformer, 通过表情和脑电两个模态来识别情绪。

## 可用的data

eeg：全
faces 缺失：s03 s04 s05 s11 s14 s15

## 问题

### 提取关键帧

不到40个视频的情况：
    - s03 s05 s14 只有780图片 缺少40.avi
    - s11 只有1-37视频

### mediapipe

没识别到脸
    - s04 photo/323照片 手挡住了脸 在第17个视频
    - s15 photo/237照片 手挡住了脸
