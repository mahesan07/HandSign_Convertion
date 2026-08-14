"""Draw one SVG hand illustration per letter, from this project's own dataset.

Every sign in ``dataset/*.csv`` is 500 recordings of 21 hand landmarks.  Taking
the per-coordinate median of those recordings gives a clean, representative
pose for the letter, which we render as a hand skeleton.

Doing it this way means the reference images in the UI are:

* **accurate** -- they are literally the signs this model was trained on, not
  a generic alphabet chart that might disagree with the classifier;
* **ours** -- generated from data the project collected, with no third-party
  image licensing to worry about;
* **reproducible** -- rerun this script after collecting new data and the
  pictures follow.

Usage::

    python -m ml.scripts.generate_sign_assets            # all letters
    python -m ml.scripts.generate_sign_assets A B C      # just these

The SVGs use a gradient that stays legible on both light and dark surfaces, so
the frontend needs no per-theme variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median
from typing import Dict, List, Sequence, Tuple

from ml.landmarks import FINGERTIPS, HAND_CONNECTIONS, NUM_LANDMARKS
from ml.paths import DATASET_DIR, SIGN_ASSETS_DIR

CANVAS = 128.0
PADDING = 14.0

# Chosen to stay legible on both the dark and the light surface, so the
# same file works in either theme with no filters or per-theme variants.
INK_FROM = "#8b8cf7"   # periwinkle
INK_TO = "#c084fc"     # violet
ACCENT = "#f9a8d4"     # pink, for fingertips


def load_median_pose(csv_path: Path) -> List[Tuple[float, float, float]]:
    """Per-coordinate median of every sample for one letter."""
    columns: List[List[float]] = [[] for _ in range(NUM_LANDMARKS * 3)]

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{csv_path.name} is empty")
        for row in reader:
            if len(row) < NUM_LANDMARKS * 3:
                continue
            for i in range(NUM_LANDMARKS * 3):
                try:
                    columns[i].append(float(row[i]))
                except ValueError:
                    break

    if not columns[0]:
        raise ValueError(f"{csv_path.name} has no usable rows")

    flat = [median(values) for values in columns]
    return [
        (flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])
        for i in range(NUM_LANDMARKS)
    ]


def project(
    pose: Sequence[Tuple[float, float, float]]
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Fit the pose into the canvas, keeping its aspect ratio.

    Returns the 2D points plus a 0-1 depth value per landmark (1 = nearest the
    camera), which we use to vary joint size so the hand reads as 3D.
    """
    xs = [p[0] for p in pose]
    ys = [p[1] for p in pose]
    zs = [p[2] for p in pose]

    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    scale = (CANVAS - 2 * PADDING) / span

    # Centre the drawing rather than anchoring it to the bounding box corner.
    offset_x = (CANVAS - (max(xs) - min(xs)) * scale) / 2 - min(xs) * scale
    offset_y = (CANVAS - (max(ys) - min(ys)) * scale) / 2 - min(ys) * scale
    points = [(x * scale + offset_x, y * scale + offset_y) for x, y in zip(xs, ys)]

    z_span = max(zs) - min(zs)
    if z_span < 1e-6:
        depths = [0.5] * len(zs)
    else:
        # MediaPipe z grows away from the camera, so invert it.
        depths = [1.0 - (z - min(zs)) / z_span for z in zs]
    return points, depths


def render_svg(letter: str, pose: Sequence[Tuple[float, float, float]]) -> str:
    points, depths = project(pose)
    gid = f"ink{letter}"

    bones: List[str] = []
    for start, end in HAND_CONNECTIONS:
        x1, y1 = points[start]
        x2, y2 = points[end]
        bones.append(
            f'    <line x1="{x1:.2f}" y1="{y1:.2f}" '
            f'x2="{x2:.2f}" y2="{y2:.2f}" />'
        )

    joints: List[str] = []
    for index, ((x, y), depth) in enumerate(zip(points, depths)):
        is_tip = index in FINGERTIPS
        radius = (4.4 if is_tip else 3.0) + depth * 1.3
        fill = ACCENT if is_tip else f"url(#{gid})"
        joints.append(
            f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
            f'fill="{fill}" />'
        )

    # The wrist gets an anchor ring so the hand's orientation is obvious.
    wrist_x, wrist_y = points[0]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" \
role="img" aria-label="Hand sign for the letter {letter}">
  <title>Letter {letter}</title>
  <defs>
    <linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="{INK_FROM}" />
      <stop offset="100%" stop-color="{INK_TO}" />
    </linearGradient>
  </defs>
  <g stroke="url(#{gid})" stroke-width="5.2" stroke-linecap="round" \
stroke-linejoin="round" fill="none" opacity="0.92">
{chr(10).join(bones)}
  </g>
{chr(10).join(joints)}
  <circle cx="{wrist_x:.2f}" cy="{wrist_y:.2f}" r="8.5" fill="none" \
stroke="url(#{gid})" stroke-width="1.6" opacity="0.45" />
</svg>
"""


def generate(letters: Sequence[str] | None = None) -> Dict[str, str]:
    """Write ``frontend/public/signs/<LETTER>.svg`` for each letter."""
    SIGN_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    available = sorted(p.stem.upper() for p in DATASET_DIR.glob("*.csv"))
    targets = [l.upper() for l in letters] if letters else available

    manifest: Dict[str, str] = {}
    for letter in targets:
        csv_path = DATASET_DIR / f"{letter}.csv"
        if not csv_path.exists():
            print(f"  skip {letter}: no dataset/{letter}.csv", file=sys.stderr)
            continue
        pose = load_median_pose(csv_path)
        out_path = SIGN_ASSETS_DIR / f"{letter}.svg"
        out_path.write_text(render_svg(letter, pose), encoding="utf-8")
        manifest[letter] = f"{letter}.svg"
        print(f"  {letter} -> {out_path.relative_to(SIGN_ASSETS_DIR.parent.parent)}")

    manifest_path = SIGN_ASSETS_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "description": (
                    "Generated by ml/scripts/generate_sign_assets.py from the "
                    "median landmark pose of each letter in dataset/."
                ),
                "signs": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "letters", nargs="*", help="letters to regenerate (default: all)"
    )
    args = parser.parse_args()

    print(f"Writing sign illustrations to {SIGN_ASSETS_DIR}")
    manifest = generate(args.letters or None)
    print(f"Done: {len(manifest)} signs.")


if __name__ == "__main__":
    main()
