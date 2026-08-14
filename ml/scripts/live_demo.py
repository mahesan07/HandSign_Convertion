"""Offline recognition demo in an OpenCV window.

This is the descendant of the original ``live_prediction.py`` and exists for
one job: proving the ML pipeline works without the backend, the browser or
Gemini being involved.  If a sign is not recognised here, the problem is in
the model, not in the web stack.

    python -m ml.scripts.live_demo
    python -m ml.scripts.live_demo --camera 1 --raw

Keys:  Q quit   SPACE finish word   BACKSPACE delete   C clear

Unlike the original script it also runs the real stabilizer and text buffer,
so holding a sign for two seconds types one letter rather than thirty.
"""

from __future__ import annotations

import argparse

from backend.app.core.config import get_settings
from backend.app.services.stabilizer import (
    PredictionStabilizer,
    RecognitionStatus,
    StabilizerConfig,
)
from backend.app.services.text_buffer import TextBuffer
from ml.landmarks import HAND_CONNECTIONS
from ml.recognizer import SignRecognizer

GREEN = (120, 220, 130)
AMBER = (80, 190, 250)
RED = (110, 110, 245)
WHITE = (245, 245, 245)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="show every frame's prediction without stabilisation",
    )
    args = parser.parse_args()

    import cv2

    settings = get_settings()
    stabilizer = PredictionStabilizer(
        StabilizerConfig(
            min_confidence=settings.min_confidence,
            partial_confidence=settings.partial_confidence,
            stable_frames=settings.stable_frames,
            decay=settings.prediction_decay,
            commit_cooldown_ms=settings.commit_cooldown_ms,
            duplicate_suppression_ms=settings.duplicate_suppression_ms,
            release_frames=settings.release_frames,
        )
    )
    buffer = TextBuffer()

    print("Loading model ...")
    recognizer = SignRecognizer()
    print(f"Ready: {len(recognizer.classes)} classes "
          f"({''.join(recognizer.classes)})")

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        recognizer.close()
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Could not read a frame from the camera.")
                break

            # Mirror, exactly as the data-collection script did -- the model
            # was trained on mirrored frames.
            frame = cv2.flip(frame, 1)
            result = recognizer.recognize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            height, width, _ = frame.shape
            if result.hand_detected:
                points = [
                    (int(x * width), int(y * height))
                    for x, y in result.landmarks_xy
                ]
                for start, end in HAND_CONNECTIONS:
                    cv2.line(frame, points[start], points[end], GREEN, 2)
                for point in points:
                    cv2.circle(frame, point, 4, WHITE, -1)

            update = stabilizer.update(result.letter, result.confidence)
            if not args.raw and update.committed_letter:
                buffer.add_character(update.committed_letter)

            if result.hand_detected:
                colour = GREEN if update.status is RecognitionStatus.COMMITTED else AMBER
                cv2.putText(
                    frame,
                    f"{result.letter}  {result.confidence * 100:.0f}%",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3,
                )
                cv2.putText(
                    frame,
                    f"{update.status.value}  {update.stable_count}/{update.required_frames}",
                    (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2,
                )
                bar = int(update.progress * 260)
                cv2.rectangle(frame, (20, 98), (280, 108), (70, 70, 70), -1)
                if bar:
                    cv2.rectangle(frame, (20, 98), (20 + bar, 108), colour, -1)
            else:
                cv2.putText(
                    frame, "No hand detected", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2,
                )

            cv2.putText(
                frame, f"{result.elapsed_ms:.0f} ms/frame",
                (width - 170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1,
            )
            cv2.rectangle(frame, (0, height - 60), (width, height), (28, 28, 32), -1)
            cv2.putText(
                frame, buffer.text[-46:] or "(nothing yet)",
                (20, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2,
            )

            cv2.imshow("HandSign Conversion - recognition check", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                buffer.space()
            elif key == 8:              # backspace
                buffer.backspace()
            elif key == ord("c"):
                buffer.clear()
                stabilizer.reset()
    finally:
        capture.release()
        cv2.destroyAllWindows()
        recognizer.close()

    print(f"\nFinal text: {buffer.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
