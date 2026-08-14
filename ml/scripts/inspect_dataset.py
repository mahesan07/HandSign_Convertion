"""Report on the whole dataset: balance, duplicates, missing values, shape.

The original version only ever looked at ``dataset/A.csv``; this one covers
every class and flags the problems that actually affect training.

    python -m ml.scripts.inspect_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ml.paths import DATASET_DIR

EXPECTED_FEATURES = 63


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    import pandas as pd

    files = sorted(args.dataset.glob("*.csv"))
    if not files:
        print(f"No CSV files in {args.dataset}")
        return 1

    print(f"{'letter':<8}{'rows':>8}{'cols':>7}{'dupes':>8}{'nulls':>8}  issues")
    print("-" * 62)

    frames = []
    problems: list[str] = []

    for path in files:
        frame = pd.read_csv(path)
        frames.append(frame)

        issues = []
        if "label" not in frame.columns:
            issues.append("no 'label' column")
        elif frame["label"].nunique() != 1:
            issues.append(f"mixed labels {sorted(frame['label'].unique())}")
        elif frame["label"].iloc[0] != path.stem.upper():
            issues.append(f"label {frame['label'].iloc[0]!r} != filename")

        feature_columns = len(frame.columns) - 1
        if feature_columns != EXPECTED_FEATURES:
            issues.append(f"{feature_columns} features, expected {EXPECTED_FEATURES}")

        nulls = int(frame.isnull().sum().sum())
        if nulls:
            issues.append(f"{nulls} missing values")

        print(
            f"{path.stem:<8}{len(frame):>8}{len(frame.columns):>7}"
            f"{int(frame.duplicated().sum()):>8}{nulls:>8}  "
            f"{'; '.join(issues) if issues else 'ok'}"
        )
        problems.extend(f"{path.name}: {issue}" for issue in issues)

    data = pd.concat(frames, ignore_index=True)
    counts = data["label"].value_counts()

    print("-" * 62)
    print(f"classes : {counts.size}")
    print(f"samples : {len(data)}")
    print(f"balance : min {counts.min()} ({counts.idxmin()}) / "
          f"max {counts.max()} ({counts.idxmax()})")

    if counts.max() > counts.min() * 1.5:
        print(
            f"\nImbalance: {counts.idxmax()} has {counts.max() / counts.min():.1f}x "
            f"the samples of {counts.idxmin()}. The forest will predict "
            f"{counts.idxmax()} more readily than it should; trimming the extra "
            "rows or collecting more of the others would even it out."
        )

    if problems:
        print("\nIssues found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\nNo structural problems found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
