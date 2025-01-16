import os
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
# 初始化MediaPipe模块
mp_face_mesh = mp.solutions.face_mesh

def extract_features(image_path):
    # 读取图片并转换为RGB格式
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        return None

    try:
        image = Image.open(image_path)
        rgb_image = image.convert('RGB')
        rgb_image = np.array(rgb_image)
    except Exception as e:
        print(f"Failed to read image: {image_path}, Error: {e}")
        return None

    # 使用MediaPipe Face Mesh检测面部特征点
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(rgb_image)

        # 提取特征点坐标
        features = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    features.append(landmark.x)
                    features.append(landmark.y)
                    features.append(landmark.z)
        else:
            print(f"No face landmarks detected in image: {image_path}")

    return features


def main():
    dataset_dir = r'E:\OneDrive\Desktop\大三上\python程序设计\face_expression_recognition\dataset'  # 数据集目录
    save_dir = os.path.abspath(r'E:\OneDrive\Desktop\大三上\python程序设计\face_expression_recognition\data')  # 保存数据的目录
    os.makedirs(save_dir, exist_ok=True)  # 确保保存目录存在
    features = []
    labels = []

    # 遍历数据集中的每个文件夹（表情类别）
    for label, folder in enumerate(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            # 遍历文件夹中的每张图片
            for image_file in os.listdir(folder_path):
                if image_file.lower().endswith('.tiff') or image_file.lower().endswith('.tif'):
                    image_path = os.path.join(folder_path, image_file)
                    feature = extract_features(image_path)
                    if feature:  # 如果成功提取到特征
                        features.append(feature)
                        labels.append(label)

    # 将特征和标签保存为NumPy数组
    if features and labels:
        features = np.array(features)
        labels = np.array(labels)
        np.save(os.path.join(save_dir, 'features.npy'), features)
        np.save(os.path.join(save_dir, 'labels.npy'), labels)
        print("Features and labels saved successfully.")
    else:
        print("No features or labels to save.")

if __name__ == '__main__':
    main()
