import cv2
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

TIP_IDS = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)
cv2.namedWindow("Two-Hand Finger Counter", cv2.WINDOW_NORMAL)

def get_gesture(fingers):
    if fingers == [0,0,0,0,0]:
        return "FIST"
    if fingers == [1,0,0,0,0]:
        return "THUMBS UP"
    if fingers == [0,1,1,0,0]:
        return "PEACE"
    if fingers == [0,1,1,1,0]:
        return "THREE"
    if fingers == [0,1,1,1,1]:
        return "FOUR"
    if fingers == [1,1,1,1,1]:
        return "OPEN PALM"
    return None   # ❗ no UNKNOWN

def draw_text(img, text, x, y, color, scale=0.7):
    # shadow
    cv2.putText(img, text, (x+2, y+2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 2)
    # main text
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    total_fingers = 0
    gestures = []
    y_pos = 35

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_info in zip(
            result.multi_hand_landmarks,
            result.multi_handedness
        ):
            hand_label = hand_info.classification[0].label
            h, w, _ = img.shape
            lm = [(int(l.x*w), int(l.y*h)) for l in hand_landmarks.landmark]

            fingers = [0]*5

            # Thumb detection (stable)
            if hand_label == "Right":
                if lm[4][0] < lm[3][0] and lm[4][0] < lm[2][0]:
                    fingers[0] = 1
            else:
                if lm[4][0] > lm[3][0] and lm[4][0] > lm[2][0]:
                    fingers[0] = 1

            # Other fingers
            for i in range(1,5):
                if lm[TIP_IDS[i]][1] < lm[TIP_IDS[i]-2][1]:
                    fingers[i] = 1

            count = sum(fingers)
            total_fingers += count
            gesture = get_gesture(fingers)

            # Store gesture only if valid
            if gesture:
                gestures.append(gesture)

            # UI
            draw_text(img, f"{hand_label}: {count} fingers",
                      20, y_pos, (0, 255, 0), 0.75)
            y_pos += 30

            if gesture:
                draw_text(img, f"{hand_label} Gesture: {gesture}",
                          20, y_pos, (255, 200, 0), 0.65)
                y_pos += 35

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ✅ TWO THUMBS UP (clean & stable)
    if len(gestures) == 2 and gestures.count("THUMBS UP") == 2:
        draw_text(img, "TWO THUMBS UP ",
                  20, y_pos, (0, 255, 255), 0.9)
        y_pos += 40

    if total_fingers > 0:
        draw_text(img, f"Total Fingers: {total_fingers}",
                  20, y_pos, (255, 255, 255), 0.8)

    cv2.imshow("Two-Hand Finger Counter", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty(
        "Two-Hand Finger Counter", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
