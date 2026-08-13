import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_features


MODEL_PATH = "models/hand_landmarker.task"


# ---------------------------------------
# MediaPipe setup
# ---------------------------------------

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ---------------------------------------
# Camera
# ---------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()


while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame")
        break

    # Mirror image
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert to MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect hand
    result = detector.detect(mp_image)


    # ---------------------------------------
    # Feature extraction
    # ---------------------------------------

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        features = extract_features(hand)

        print(
            "Features:",
            len(features),
            "| First 6:",
            features[:6]
        )


        # Draw landmarks
        h, w, _ = frame.shape

        for landmark in hand:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        cv2.putText(
            frame,
            "Hand detected | 63 features",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


    cv2.imshow(
        "Live Feature Test",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()