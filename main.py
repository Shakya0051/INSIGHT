import cv2
from ultralytics import YOLO
import mediapipe as mp
import pyttsx3
import threading
import time

# ---------------- SPEECH ENGINE ----------------
engine = pyttsx3.init()
engine.setProperty('rate',150)

def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run).start()

# ---------------- YOLO MODEL ----------------
model = YOLO("yolov8n.pt")

# ---------------- MEDIAPIPE HANDS ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- CAMERA ----------------
cap = cv2.VideoCapture(0)

detect_enabled = True
last_message = ""

cooldown = 5
last_spoken = {}

gesture_delay = 2
last_gesture_time = 0

# ---------------- GESTURE FUNCTION ----------------
def detect_gesture(hand):

    thumb_tip = hand.landmark[4]
    thumb_ip = hand.landmark[3]

    index_tip = hand.landmark[8]
    index_pip = hand.landmark[6]

    middle_tip = hand.landmark[12]
    middle_pip = hand.landmark[10]

    ring_tip = hand.landmark[16]
    ring_pip = hand.landmark[14]

    pinky_tip = hand.landmark[20]
    pinky_pip = hand.landmark[18]

    thumb_up = thumb_tip.y < thumb_ip.y
    index_up = index_tip.y < index_pip.y
    middle_up = middle_tip.y < middle_pip.y
    ring_up = ring_tip.y < ring_pip.y
    pinky_up = pinky_tip.y < pinky_pip.y

    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "thumbs_up"

    if thumb_up and index_up and middle_up and ring_up and pinky_up:
        return "open"

    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"

    return "none"

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # -------- HAND GESTURES --------
    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        gesture = detect_gesture(hand)
        current_time = time.time()

        if current_time - last_gesture_time > gesture_delay:

            if gesture == "thumbs_up":
                if not detect_enabled:
                    detect_enabled = True
                    speak("Detection started")

            elif gesture == "open":
                if detect_enabled:
                    detect_enabled = False
                    speak("Detection paused")

            elif gesture == "fist":
                if last_message != "":
                    speak(last_message)

            last_gesture_time = current_time

    # -------- OBJECT DETECTION --------
    if detect_enabled:

        results = model(frame)
        current_time = time.time()

        for r in results:

            boxes = r.boxes

            if boxes is not None:

                for box in boxes:

                    x1,y1,x2,y2 = box.xyxy[0]

                    center_x = int((x1+x2)/2)

                    cls = int(box.cls[0])
                    name = model.names[cls]

                    if center_x < width/3:
                        direction = "on the left"
                    elif center_x < 2*width/3:
                        direction = "ahead"
                    else:
                        direction = "on the right"

                    message = f"{name} {direction}"
                    last_message = message

                    if message not in last_spoken or current_time-last_spoken[message] > cooldown:

                        print("Detected:",message)
                        speak(message)

                        last_spoken[message] = current_time

    cv2.imshow("Blind Assistance Smart Glass Prototype", frame)

    if cv2.waitKey(1)==27:
        break

    time.sleep(1)

cap.release()
cv2.destroyAllWindows()