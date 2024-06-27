import cv2
import mediapipe as mp
# 进度条库
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt


# 导入mediapipe人脸检测模型
mp_face_detection = mp.solutions.face_detection
model = mp_face_detection.FaceDetection(   
        min_detection_confidence=0.7, # 置信度阈值，过滤掉小于置信度的预测框
        model_selection=0,            # 选择模型，0适用于人脸距离镜头近（2米以内），1适用于人脸距离镜头远（5米以内）
)

def process_participant_data(participant_id):
    participant_str = f's{participant_id:02}'
    input_dir = f'./photo/{participant_str}/'
    output_dir = f'./faces/{participant_str}/'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = os.listdir(input_dir)
    valid_image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    # 遍历图片文件
    for image_file in image_files:
        # 获取文件扩展名
        filename, ext = os.path.splitext(image_file)
        # 如果文件扩展名不在有效的图像文件扩展名列表中，跳过此文件
        if ext.lower() not in valid_image_extensions:
            continue

        # 从图片文件读入图像，opencv读入为BGR格式
        img = cv2.imread(f'{input_dir}{image_file}')
        if img is None:
            print(f"Failed to load image at '{input_dir}{image_file}'")
            continue
        else:
            print(f"'{input_dir}{image_file}'")

        # BGR转RGB
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 将RGB图像输入模型，获取预测结果
        results = model.process(img_RGB)

        # 可视化人脸框和人脸关键点
        annotated_image = img.copy()

        # 取结果中最像人脸的一张
        detection = results.detections[0]
        
        # 获取人脸框的坐标
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        bbox = [int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)]
        
        # 计算新的人脸框的坐标，扩大10%的距离。同时避免扩大到图像外面
        expansion_ratio = 0.1
        bbox_expanded = [max(0, bbox[0] - bbox[2] * expansion_ratio),  # 左边
                    max(0, bbox[1] - bbox[3] * expansion_ratio),  # 上边
                    min(iw, bbox[0] + bbox[2] * (1 + expansion_ratio)),  # 右边
                    min(ih, bbox[1] + bbox[3] * (1 + expansion_ratio))]  # 下边

        # 使用新的坐标来绘制人脸框
        cv2.rectangle(annotated_image, (int(bbox_expanded[0]), int(bbox_expanded[1])), (int(bbox_expanded[2]), int(bbox_expanded[3])), (255,0,0), 2)

        # 切出人脸部分的图像
        face_image = img[int(bbox_expanded[1]):int(bbox_expanded[3]), int(bbox_expanded[0]):int(bbox_expanded[2])]
        
        # 保存切出人脸部分的图像到faceFromFrame文件夹
        cv2.imwrite('{}{}{}'.format(output_dir, filename, ext), face_image)

# 循环处理s01到s22的数据
for participant_id in range(18, 23):
    process_participant_data(participant_id)