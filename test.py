import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 加载预训练模型
try:
    model_path = 'models/expression_recognition_model.pkl'
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {os.path.abspath(model_path)}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 表情标签映射
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def preprocess_landmarks(landmarks):
    """将MediaPipe提取的关键点转换为适合模型输入的格式"""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return
    else:
        print("Video capture device opened successfully.")

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

            detected_emotion = "neutral"
            emotion_probabilities = None
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 绘制面部关键点
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    landmarks = preprocess_landmarks(face_landmarks.landmark)
                    print(f"Detected landmarks: {landmarks[:10]}")  # Print first 10 landmarks for brevity
                    # 使用模型预测情绪
                    prediction = model.predict_proba([landmarks])[0]
                    emotion_idx = np.argmax(prediction)
                    detected_emotion = emotion_labels[emotion_idx]
                    emotion_probabilities = prediction
                    print(f"Detected emotion: {detected_emotion} with probabilities: {prediction}")

            # 在帧上显示检测到的情绪和概率
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 255, 255)  # White color
            line_type = 2

            text = f"Emotion: {detected_emotion}"
            position = (10, 30)
            cv2.putText(frame_bgr, text, position, font, font_scale, font_color, line_type)

            if emotion_probabilities is not None:
                y_offset = 60
                for i, label in enumerate(emotion_labels):
                    prob_text = f"{label}: {emotion_probabilities[i]:.2f}"
                    prob_position = (10, 30 + y_offset * i)
                    cv2.putText(frame_bgr, prob_text, prob_position, font, font_scale, font_color, line_type)

            # 显示带有关键点的帧
            cv2.imshow('Face Detection', frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()



