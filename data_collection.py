import cv2
import mediapipe as mp
import csv
import os
import time

from features import extract_features


LABEL = input("Enter sign label (A-Z): ").strip().upper()

if len(LABEL) != 1 or not LABEL.isalpha():
    print("ERROR: Please enter exactly one letter from A-Z.")
    exit()

TARGET_SAMPLES = 500
SAMPLES_PER_SECOND = 5

DATASET_DIR = "dataset"
CSV_FILE = os.path.join(DATASET_DIR, f"{LABEL}.csv")

SAMPLE_INTERVAL = 1.0 / SAMPLES_PER_SECOND


# ==========================================
# MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)


# ==========================================
# CSV SETUP
# ==========================================

os.makedirs(DATASET_DIR, exist_ok=True)

file_exists = os.path.exists(CSV_FILE)

csv_file = open(
    CSV_FILE,
    "a",
    newline=""
)

writer = csv.writer(csv_file)

if not file_exists:
    writer.writerow(
        [f"feature_{i}" for i in range(63)] + ["label"]
    )


# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    csv_file.close()
    landmarker.close()
    exit()


# ==========================================
# GET EXISTING SAMPLE COUNT
# ==========================================

if file_exists:
    with open(CSV_FILE, "r", newline="") as f:
        existing_rows = sum(1 for row in f) - 1
        sample_count = max(existing_rows, 0)
else:
    sample_count = 0

recording = False
countdown = False

countdown_start = 0
last_sample_time = 0


print("================================")
print("SIGN DATA COLLECTION")
print("================================")
print(f"Label: {LABEL}")
print(f"Existing samples: {sample_count}")
print(f"Target samples: {TARGET_SAMPLES}")
print(f"Remaining samples: {max(TARGET_SAMPLES - sample_count, 0)}")
print(f"Sampling rate: {SAMPLES_PER_SECOND} samples/sec")
print()
print("Press R to start recording")
print("Press Q to quit")
print()


# ==========================================
# MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    # --------------------------------------
    # MIRROR CAMERA
    # --------------------------------------

    frame = cv2.flip(frame, 1)

    # --------------------------------------
    # CONVERT BGR -> RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------
    # CREATE MEDIAPIPE IMAGE
    # --------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # --------------------------------------
    # DETECT HAND
    # --------------------------------------

    result = landmarker.detect(mp_image)

    # --------------------------------------
    # DRAW LANDMARKS
    # --------------------------------------

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        # Draw connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        height, width, _ = frame.shape

        # Draw points
        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections
        for start, end in connections:

            x1 = int(hand_landmarks[start].x * width)
            y1 = int(hand_landmarks[start].y * height)

            x2 = int(hand_landmarks[end].x * width)
            y2 = int(hand_landmarks[end].y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    # ======================================
    # COUNTDOWN
    # ======================================

    if countdown:

        elapsed = time.time() - countdown_start

        remaining = 3 - int(elapsed)

        if remaining > 0:

            cv2.putText(
                frame,
                f"Starting in {remaining}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                3
            )

        else:

            countdown = False
            recording = True

            last_sample_time = time.time()

            print("RECORDING STARTED")

    # ======================================
    # RECORDING
    # ======================================

    if recording:

        current_time = time.time()

        # Time to collect next sample?
        if current_time - last_sample_time >= SAMPLE_INTERVAL:

            if result.hand_landmarks:

                hand_landmarks = result.hand_landmarks[0]

                # Use YOUR existing feature extractor
                features = extract_features(
                    hand_landmarks
                )

                writer.writerow(
                    features + [LABEL]
                )

                csv_file.flush()

                sample_count += 1

                last_sample_time = current_time

                print(
                    f"Sample: {sample_count} / {TARGET_SAMPLES}"
                )

            else:

                cv2.putText(
                    frame,
                    "HAND NOT DETECTED",
                    (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        # Progress
        cv2.putText(
            frame,
            f"Recording: {sample_count}/{TARGET_SAMPLES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Target reached
        if sample_count >= TARGET_SAMPLES:

            recording = False

            print()
            print("================================")
            print("TARGET REACHED")
            print(f"Collected: {sample_count}")
            print(f"Saved to: {CSV_FILE}")
            print("================================")

    else:

        if not countdown:

            cv2.putText(
                frame,
                "Press R to start recording",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

    # ======================================
    # SHOW CAMERA
    # ======================================

    cv2.imshow(
        "Sign Data Collection",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    # Start recording
    if (
    key == ord("r") and not recording and not countdown and sample_count < TARGET_SAMPLES ):
    
        countdown = True
        countdown_start = time.time()

        print()
        print("Starting in 3 seconds...")

    # Quit
    elif key == ord("q"):

        print("Exiting...")
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()
cv2.destroyAllWindows()

landmarker.close()
csv_file.close()

print("Done.")