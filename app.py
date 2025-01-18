import os
import base64
import random

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from threading import Lock

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

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
emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 表情映射到对应的图片
emoji_mapping = {
    "angry": "angry.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "neutral": "neutral.png",
    "sad": "sad.png",
    "surprise": "surprise.png"
}

current_emoji_index = 2
lock = Lock()  # 创建一个锁对象用于线程同步
emojis = list(emoji_mapping.keys())


def preprocess_landmarks(landmarks, _scaler):
    """将MediaPipe提取的关键点转换为适合模型输入的格式，并进行标准化"""
    landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return _scaler.transform([landmarks_array])[0]


@socketio.on('connect')
def handle_connect():
    global current_emoji_index
    with lock:  # 使用锁来保护对共享资源的访问
        current_emoji_index = 2
        next_emoji = emoji_mapping[emojis[current_emoji_index]]
        required_emotion = emojis[current_emoji_index]
        emit('update_emoji', {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})
        print(f"Client connected and update_emoji emitted: {next_emoji}, {required_emotion}")


@socketio.on('send_frame')
def handle_frame(data):
    try:
        encoded_data = data['frame'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = preprocess_landmarks(face_landmarks.landmark, scaler)  # 传入scaler
                prediction = model.predict([landmarks])[0]
                if prediction > 5:
                    break
                detected_emotion = emotion_labels[prediction]
                socketio.emit('detected_emotion', {'detectedEmotion': detected_emotion})

                global current_emoji_index  # 明确指出我们使用的是全局变量
                with lock:  # 使用锁来保护对共享资源的访问
                    if detected_emotion == emojis[current_emoji_index]:
                        current_emoji_index = current_emoji_index = random.randint(0, len(emojis) - 1)
                        next_emoji = emoji_mapping[emojis[current_emoji_index]]
                        required_emotion = emojis[current_emoji_index]
                        socketio.emit('update_emoji', {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})
    except Exception as e:
        print(f"Error processing frame: {e}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture device.")
        return

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detected_emotion = "unknown"
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # 【引用】使用新的预处理函数，并传入scaler进行标准化
                    landmarks = preprocess_landmarks(face_landmarks.landmark, scaler)  # 传入scaler
                    prediction = model.predict([landmarks])[0]
                    if prediction > 5:
                        break
                    detected_emotion = emotion_labels[prediction]
                    print(f"Detected emotion: {detected_emotion}")

                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )

                    socketio.emit('detected_emotion', {'detectedEmotion': detected_emotion})

                    with lock:
                        if detected_emotion == emojis[current_emoji_index]:
                            current_emoji_index = random.randint(0, len(emojis) - 1)
                            next_emoji = emoji_mapping[emojis[current_emoji_index]]
                            required_emotion = emojis[current_emoji_index]
                            socketio.emit('update_emoji',
                                          {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame_base64 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_base64 + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()

@socketio.on('give_up')
def handle_give_up():
    global current_emoji_index
    with lock:
        current_emoji_index = random.randint(0, len(emojis) - 1)
        next_emoji = emoji_mapping[emojis[current_emoji_index]]
        required_emotion = emojis[current_emoji_index]
        socketio.emit('update_emoji',
                      {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})
        print(f"Give up button clicked, update_emoji emitted: {next_emoji}, {required_emotion}")
        print(f"Give up button clicked, update_emoji emitted: {next_emoji}, {required_emotion}")

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
