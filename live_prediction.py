import cv2
import mediapipe as mp
import joblib

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from features import extract_features


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/hand_landmarker.task"
CLASSIFIER_PATH = "sign_model.pkl"


# ==========================================
# LOAD ML MODEL
# ==========================================

model = joblib.load(CLASSIFIER_PATH)

print("Model loaded successfully!")
print("Classes:", model.classes_)


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

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


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Cannot open camera")

    detector.close()

    exit()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Cannot read frame")

        break


    # Mirror camera
    frame = cv2.flip(frame, 1)


    # BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # Detect hand
    result = detector.detect(mp_image)


    # ======================================
    # HAND DETECTED
    # ======================================

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]


        # ----------------------------------
        # Extract same 63 features
        # ----------------------------------

        features = extract_features(hand)


        # ----------------------------------
        # Predict letter
        # ----------------------------------

        prediction = model.predict(
            [features]
        )[0]


        # ----------------------------------
        # Prediction probability
        # ----------------------------------

        probabilities = model.predict_proba(
            [features]
        )[0]

        confidence = max(probabilities)


        # ----------------------------------
        # Draw landmarks
        # ----------------------------------

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


        # ----------------------------------
        # Display prediction
        # ----------------------------------

        text = f"{prediction} ({confidence * 100:.1f}%)"

        cv2.putText(
            frame,
            text,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )


    else:

        cv2.putText(
            frame,
            "No hand detected",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )


    # Show camera
    cv2.imshow(
        "Real-Time Sign Translator",
        frame
    )


    # Q → quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

detector.close()