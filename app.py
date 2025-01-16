import cv2
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


# 定义情绪阈值（这里是一个稍微改进的例子）
def get_emotion(landmarks):
    if len(landmarks) < 468:
        return "未检测到足够的关键点"

    # 获取眼睛和嘴巴的关键点位置
    left_eye_openness = abs(landmarks[145].y - landmarks[159].y)
    right_eye_openness = abs(landmarks[374].y - landmarks[386].y)
    mouth_openness = abs(landmarks[13].y - landmarks[14].y)
    eye_avg_openness = (left_eye_openness + right_eye_openness) / 2

    # 计算眉毛提升程度
    left_eyebrow_lift = abs(landmarks[105].y - landmarks[6].y)
    right_eyebrow_lift = abs(landmarks[336].y - landmarks[296].y)
    eyebrow_avg_lift = (left_eyebrow_lift + right_eyebrow_lift) / 2

    # 计算嘴角上扬程度
    left_mouth_upturn = abs(landmarks[291].y - landmarks[0].y)
    right_mouth_upturn = abs(landmarks[61].y - landmarks[0].y)
    mouth_upturn_avg = (left_mouth_upturn + right_mouth_upturn) / 2

    # 计算嘴角向下拉的程度
    left_mouth_downward = abs(landmarks[17].y - landmarks[0].y)
    right_mouth_downward = abs(landmarks[269].y - landmarks[0].y)
    mouth_downward_avg = (left_mouth_downward + right_mouth_downward) / 2

    # 调试信息
    debug_info = {
        "eye_avg_openness": eye_avg_openness,
        "eyebrow_avg_lift": eyebrow_avg_lift,
        "mouth_openness": mouth_openness,
        "mouth_upturn_avg": mouth_upturn_avg,
        "mouth_downward_avg": mouth_downward_avg
    }

    # 情绪判断逻辑
    if eyebrow_avg_lift > 0.02 and eye_avg_openness < 0.01:
        emotion = "angry"
    elif eye_avg_openness > 0.02 and mouth_openness > 0.02:
        emotion = "surprise"
    elif mouth_upturn_avg > 0.02:
        emotion = "happy"
    elif mouth_downward_avg > 0.02:
        emotion = "sad"
    elif eye_avg_openness > 0.01:
        emotion = "fear"
    else:
        emotion = "neutral"

    return emotion, debug_info


# 加载中文字体
try:
    font_path = "simhei.ttf"  # 确保你有一个可用的中文字体文件
    font = ImageFont.truetype(font_path, 30)
except IOError:
    print("字体文件不存在或路径错误，请确保 simhei.ttf 文件存在")
    font = ImageFont.load_default()

# 表情映射
emoji_mapping = {
    "angry": "angry.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "sad": "sad.png",
    "surprise": "surprised.png",
    "neutral": "neutral.png"
}

current_emoji_index = 0
emojis = list(emoji_mapping.keys())


@socketio.on('connect')
def handle_connect():
    global current_emoji_index
    current_emoji_index = 0
    emit('update_emoji',
         {'nextEmoji': emoji_mapping[emojis[current_emoji_index]], 'requiredEmotion': emojis[current_emoji_index]})


def generate_frames():
    global current_emoji_index
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            detected_emotion = "neutral"
            debug_info = {}
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    emotion, debug_info = get_emotion(face_landmarks.landmark)
                    pil_image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((50, 50), emotion, font=font, fill=(0, 255, 0))
                    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    detected_emotion = emotion

                    if emotion == emojis[current_emoji_index]:
                        current_emoji_index = (current_emoji_index + 1) % len(emojis)
                        socketio.emit('update_emoji', {'nextEmoji': emoji_mapping[emojis[current_emoji_index]],
                                                       'requiredEmotion': emojis[current_emoji_index]})

                    # 发送当前检测到的情绪给前端
                    socketio.emit('detected_emotion', {'detectedEmotion': detected_emotion, 'debugInfo': debug_info})

            ret, buffer = cv2.imencode('.jpg', frame_bgr)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)



