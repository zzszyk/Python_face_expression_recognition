import cv2
import mediapipe as mp
import numpy as np
import joblib

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 加载预训练模型和标准化器
try:
    model_path = r'models\expression_recognition_model.pkl'
    scaler_path = r'models\scaler.pkl'
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"Model and scaler loaded successfully from: {model_path} and {scaler_path}")
except Exception as e:
    print(f"Failed to load model or scaler: {e}")
    exit(1)

# 表情标签映射
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_landmarks(landmarks, scaler):
    """将MediaPipe提取的关键点转换为适合模型输入的格式，并进行标准化"""
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return scaler.transform([landmarks_array])[0]

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
    else:
        print("Video capture device opened successfully.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from camera")
                break
            else:
                print("Frame read successfully")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(frame_rgb)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                detected_emotion = "unknown"
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # 绘制面部关键点
                        mp_drawing.draw_landmarks(
                            image=frame_bgr,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                        landmarks = preprocess_landmarks(face_landmarks.landmark, scaler)
                        print(f"Detected landmarks: {landmarks[:10]}")  # Print first 10 landmarks for brevity
                        # 使用模型预测情绪
                        prediction = model.predict([landmarks])[0]
                        detected_emotion = emotion_labels[prediction]
                        print(f"Detected emotion: {detected_emotion}")

                # 在视频帧上显示检测到的表情
                cv2.putText(frame_bgr, f"Detected Emotion: {detected_emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

                # 显示带有关键点的帧
                cv2.imshow('Real-time Expression Detection', frame_bgr)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()