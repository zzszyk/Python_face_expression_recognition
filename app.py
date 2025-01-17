import os
import random
import cv2
import joblib
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import base64
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 初始化MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化绘图工具
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 加载预训练模型
try:
    model_path = r'models\expression_recognition_model.pkl'
    model = joblib.load(model_path)
    print(f"Model loaded successfully from: {os.path.abspath(model_path)}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 表情标签映射
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 表情映射到对应的图片
emoji_mapping = {
    "angry": "angry.png",
    "disgust": "disgusted.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "neutral": "neutral.png",
    "sad": "sad.png",
    "surprise": "surprised.png"
}

current_emoji_index = 0
emojis = list(emoji_mapping.keys())


def preprocess_landmarks(landmarks):
    """将MediaPipe提取的关键点转换为适合模型输入的格式"""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()


@socketio.on('connect')
def handle_connect():
    global current_emoji_index
    current_emoji_index = 0
    next_emoji = emoji_mapping[emojis[current_emoji_index]]
    required_emotion = emojis[current_emoji_index]
    emit('update_emoji', {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})
    print(f"Client connected and update_emoji emitted: {next_emoji}, {required_emotion}")


@socketio.on('send_frame')
def handle_frame(data):
    # 接收前端发送的帧数据
    frame_data = data['frame']
    import base64
    import numpy as np
    import cv2
    try:
        encoded_data = frame_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # 处理帧数据，例如进行面部关键点检测和情绪预测
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        # 之前的处理逻辑
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                landmarks = preprocess_landmarks(face_landmarks.landmark)
                prediction = model.predict([landmarks])[0]
                detected_emotion = emotion_labels[prediction]
                socketio.emit('detected_emotion', {'detectedEmotion': detected_emotion})
    except Exception as e:
        print(f"Error processing frame: {e}")


def generate_frames():
    global current_emoji_index
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

                        landmarks = preprocess_landmarks(face_landmarks.landmark)
                        print(f"Detected landmarks: {landmarks[:10]}")  # Print first 10 landmarks for brevity
                        # 使用模型预测情绪
                        prediction = model.predict([landmarks])[0]
                        detected_emotion = emotion_labels[prediction]
                        print(f"Detected emotion: {detected_emotion}")
                        # 更新当前检测到的情绪给前端
                        socketio.emit('detected_emotion', {'detectedEmotion': detected_emotion})
                        print(f"Emitting detected_emotion: {detected_emotion} to frontend")

                        if detected_emotion == emojis[current_emoji_index]:
                            current_emoji_index = (current_emoji_index + 1) % len(emojis)
                            next_emoji = emoji_mapping[emojis[current_emoji_index]]
                            required_emotion = emojis[current_emoji_index]
                            socketio.emit('update_emoji',
                                          {'nextEmoji': next_emoji, 'requiredEmotion': required_emotion})
                            print(f"Updated emoji to: {next_emoji}, {required_emotion}")

                # 显示带有关键点的帧
                ret, buffer = cv2.imencode('.jpg', frame_bgr)
                frame_base64 = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_base64 + b'\r\n')
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
