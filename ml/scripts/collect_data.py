"""Record training samples for one letter.

Unchanged in behaviour from the original ``data_collection.py``: it appends
63-feature rows to ``dataset/<LETTER>.csv`` at 5 samples/second while you hold
the sign, using the same feature extractor the classifier was trained on.

    python -m ml.scripts.collect_data            # asks for the letter
    python -m ml.scripts.collect_data --label K --target 500

Keys:  R start recording   Q quit
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from ml.detector import HandDetector
from ml.features import extract_features
from ml.landmarks import HAND_CONNECTIONS
from ml.paths import DATASET_DIR


def count_existing_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", help="the letter being recorded (A-Z)")
    parser.add_argument("--target", type=int, default=500, help="samples wanted")
    parser.add_argument("--rate", type=float, default=5.0, help="samples/second")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    import cv2

    label = (args.label or input("Enter sign label (A-Z): ")).strip().upper()
    if len(label) != 1 or not label.isalpha():
        print("Please give exactly one letter from A-Z.")
        return 1

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATASET_DIR / f"{label}.csv"
    file_existed = csv_path.exists()
    sample_count = count_existing_rows(csv_path)
    interval = 1.0 / max(args.rate, 0.1)

    detector = HandDetector()
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        detector.close()
        return 1

    handle = csv_path.open("a", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    if not file_existed:
        writer.writerow([f"feature_{i}" for i in range(63)] + ["label"])

    print("=" * 40)
    print(f"Collecting samples for: {label}")
    print(f"Already recorded      : {sample_count}")
    print(f"Target                : {args.target}")
    print(f"Rate                  : {args.rate}/second")
    print("=" * 40)
    print("Press R to start, Q to quit.")

    recording = False
    countdown_from = 0.0
    last_sample = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            frame = cv2.flip(frame, 1)
            landmarks = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            height, width, _ = frame.shape
            if landmarks:
                points = [
                    (int(lm.x * width), int(lm.y * height)) for lm in landmarks
                ]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], (120, 220, 130), 2)
                for point in points:
                    cv2.circle(frame, point, 4, (245, 245, 245), -1)

            now = time.time()

            if countdown_from:
                remaining = 3 - int(now - countdown_from)
                if remaining > 0:
                    cv2.putText(
                        frame, f"Starting in {remaining}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (80, 190, 250), 3,
                    )
                else:
                    countdown_from = 0.0
                    recording = True
                    last_sample = now
                    print("Recording ...")

            if recording:
                if now - last_sample >= interval:
                    if landmarks:
                        writer.writerow(extract_features(landmarks) + [label])
                        handle.flush()
                        sample_count += 1
                        last_sample = now
                        print(f"  {sample_count} / {args.target}", end="\r")
                    else:
                        cv2.putText(
                            frame, "HAND NOT DETECTED", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (110, 110, 245), 2,
                        )
                cv2.putText(
                    frame, f"Recording {sample_count}/{args.target}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 220, 130), 2,
                )
                if sample_count >= args.target:
                    recording = False
                    print(f"\nTarget reached: {sample_count} samples in {csv_path}")
            elif not countdown_from:
                cv2.putText(
                    frame, "Press R to record", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2,
                )

            cv2.imshow(f"Collecting sign data: {label}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not recording and not countdown_from:
                countdown_from = time.time()
                print("Starting in 3 seconds ...")
    finally:
        capture.release()
        cv2.destroyAllWindows()
        detector.close()
        handle.close()

    print(f"\nSaved {sample_count} samples to {csv_path}")
    print("Retrain with: python -m ml.scripts.train_model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
