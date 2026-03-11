from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import math

app = Flask(__name__)

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

gesture_history = []


def finger_up(hand_landmarks, tip, pip):
    return hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y


def generate_frames():
    global gesture_history

    while True:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "Detecting..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                index_up = finger_up(hand_landmarks, 8, 6)
                middle_up = finger_up(hand_landmarks, 12, 10)
                ring_up = finger_up(hand_landmarks, 16, 14)
                pinky_up = finger_up(hand_landmarks, 20, 18)

                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                h, w, c = frame.shape

                thumb_x, thumb_y = int(thumb_tip.x*w), int(thumb_tip.y*h)
                index_x, index_y = int(index_tip.x*w), int(index_tip.y*h)

                distance = math.hypot(index_x-thumb_x, index_y-thumb_y)

                if distance < 40:
                    gesture = "OK"

                elif index_up and middle_up and ring_up and pinky_up:
                    gesture = "Palm"

                elif index_up and middle_up and not ring_up and not pinky_up:
                    gesture = "Peace"

                elif index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Pointing"

                elif not index_up and not middle_up and not ring_up and not pinky_up:
                    gesture = "Fist"

                if len(gesture_history) == 10:
                    gesture_history.pop(0)

                gesture_history.append(gesture)

                cv2.putText(frame,
                            "Gesture: " + gesture,
                            (40,70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,0),
                            3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', history=gesture_history)


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)