from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')
camera = cv2.VideoCapture(0)

# بارگذاری مدل SVM
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

detected_text = ""  # متغیر برای ذخیره متن شناسایی‌شده
last_prediction = ""  # متغیر برای ذخیره آخرین پیش‌بینی شناسایی‌شده

def generate_frames():
    global detected_text, last_prediction
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # پردازش فریم
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                # گرفتن مختصات لندمارک‌ها برای مدل SVM
                landmarks = result.multi_hand_landmarks[0]
                data = [lm for lm in landmarks.landmark]
                features = np.array([[lm.x, lm.y, lm.z] for lm in data]).flatten()
                
                if len(features) == 63:  # مطمئن شوید ویژگی‌ها به درستی محاسبه شده‌اند
                    prediction = svm.predict([features])[0]

                    # اگر پیش‌بینی جدید متفاوت از آخرین پیش‌بینی باشد، آن را اضافه کن
                    if prediction != last_prediction:
                        detected_text += str(prediction)
                        last_prediction = prediction  # بروزرسانی آخرین پیش‌بینی
                else:
                    prediction = "Unknown"

                # نمایش متن تشخیص داده شده
                cv2.putText(frame, str(prediction), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    """ارسال متن شناسایی‌شده به HTML"""
    global detected_text
    return jsonify({'text': detected_text})


@app.route('/clear_text', methods=['POST'])
def clear_text():
    global detected_text
    detected_text = ""  # پاک کردن متن شناسایی‌شده
    return '', 204  # پاسخ بدون محتوا (موفقیت)



if __name__ == "__main__":
    app.run(debug=True)
