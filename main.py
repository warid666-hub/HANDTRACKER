import cv2
import mediapipe as mp
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]
history_len = 5
finger_history = deque(maxlen=history_len)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Two-Hand Finger Counter", cv2.WINDOW_NORMAL)

def get_gesture(fingers):
    if fingers == [0,0,0,0,0]: return "FIST"
    elif fingers == [1,0,0,0,0]: return "THUMBS UP"
    elif fingers == [0,1,1,0,0]: return "PEACE"
    elif fingers == [0,1,1,1,0]: return "THREE"
    elif fingers == [0,1,1,1,1]: return "FOUR"
    elif fingers == [1,1,1,1,1]: return "OPEN PALM"
    elif fingers == [1,1,0,0,1]: return "OK SIGN"
    else: return "UNKNOWN"

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    all_finger_counts = []
    all_gestures = []

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = hand_info.classification[0].label
            h, w, _ = img.shape
            lm = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks.landmark]

            fingers = [0]*5
            if hand_label == "Right":
                if lm[4][0] < lm[3][0] and lm[4][0] < lm[0][0]: fingers[0] = 1
            else:
                if lm[4][0] > lm[3][0] and lm[4][0] > lm[0][0]: fingers[0] = 1

            for i, tip_id in enumerate(TIP_IDS[1:], start=1):
                if lm[tip_id][1] < lm[tip_id-2][1] and lm[tip_id][1] < lm[tip_id-1][1]: fingers[i] = 1

            finger_history.append(fingers)
            avg_fingers = [round(sum(f[i] for f in finger_history)/len(finger_history)) for i in range(5)]

            all_finger_counts.append((hand_label, sum(avg_fingers)))
            all_gestures.append((hand_label, get_gesture(avg_fingers)))

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw UI
    y_pos = 20
    total_fingers = 0
    for hand_label, count in all_finger_counts:
        total_fingers += count
        # rectangle behind text
        cv2.rectangle(img, (10, y_pos-5), (250, y_pos+30), (0,0,0), -1)
        cv2.putText(img, f"{hand_label}: {count} fingers", (15, y_pos+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_pos += 40

    for hand_label, gesture in all_gestures:
        cv2.rectangle(img, (10, y_pos-5), (300, y_pos+30), (0,0,0), -1)
        cv2.putText(img, f"{hand_label} Gesture: {gesture}", (15, y_pos+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        y_pos += 35

    if len(all_gestures) == 2 and all(g[1]=="THUMBS UP" for g in all_gestures):
        cv2.rectangle(img, (10, y_pos-5), (350, y_pos+30), (0,0,0), -1)
        cv2.putText(img, "TWO THUMBS UP 👍👍", (15, y_pos+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        y_pos += 35

    if len(all_finger_counts) > 0:
        cv2.rectangle(img, (10, y_pos-5), (300, y_pos+30), (0,0,0), -1)
        cv2.putText(img, f"Total Fingers: {total_fingers}", (15, y_pos+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    cv2.imshow("Two-Hand Finger Counter", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty("Two-Hand Finger Counter", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
