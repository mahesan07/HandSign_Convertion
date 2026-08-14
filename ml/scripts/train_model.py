"""Train the sign classifier from ``dataset/*.csv``.

The architecture is unchanged from the original project -- a 200-tree random
forest with ``random_state=42`` -- so a plain run reproduces the shipped
``models/sign_model.pkl``.

    python -m ml.scripts.train_model                       # reproduce the original
    python -m ml.scripts.train_model --augment 5           # recommended: +3 points
    python -m ml.scripts.train_model --augment 5 --output models/candidate.pkl

You do **not** need to run this to use the app; a trained model is committed.

Two accuracies are always reported, because only one of them is meaningful:

* the **random** split is what the original script printed. Samples are
  recorded 5x/second from a held pose, so consecutive rows are near-duplicates
  and land on both sides of the split. It flatters the model by ~7 points.
* the **block** split holds out the last 20% of each recording, which is much
  closer to "the same sign, made again later". Judge the model by this one.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ml.paths import DATASET_DIR, SIGN_MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=SIGN_MODEL_PATH)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--augment",
        type=int,
        default=0,
        metavar="N",
        help=(
            "add N rotated/jittered copies of the training data. 5 is the "
            "measured sweet spot (+3 points on the block split); 0 reproduces "
            "the original model exactly."
        ),
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=1,
        help="raise to 2-4 to shrink the model file when augmenting",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="overwrite the existing model without keeping a .bak copy",
    )
    args = parser.parse_args()

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    from ml.augmentation import augment as augment_features

    files = sorted(args.dataset.glob("*.csv"))
    if not files:
        print(f"No CSV files in {args.dataset}", file=sys.stderr)
        return 1

    per_letter = {}
    for path in files:
        frame = pd.read_csv(path)
        per_letter[path.stem.upper()] = frame
        print(f"  {path.name:<10} {len(frame):>6} samples")

    data = pd.concat(per_letter.values(), ignore_index=True)
    feature_columns = [c for c in data.columns if c != "label"]
    X = data[feature_columns]
    y = data["label"]

    print()
    print(f"Total samples : {len(data)}")
    print(f"Features      : {len(feature_columns)}")
    print(f"Classes       : {y.nunique()} -> {''.join(sorted(y.unique()))}")

    counts = y.value_counts()
    if counts.max() > counts.min() * 1.5:
        print(
            f"  note: imbalanced ({counts.idxmax()}={counts.max()} vs "
            f"{counts.idxmin()}={counts.min()}); the forest will favour "
            f"{counts.idxmax()}."
        )

    def build() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            min_samples_leaf=args.min_samples_leaf,
            n_jobs=-1,
        )

    def prepare(features, labels):
        if not args.augment:
            return features, labels
        return augment_features(
            np.asarray(features), np.asarray(labels), copies=args.augment
        )

    # ---------------------------------------------------------- random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    Xa, ya = prepare(X_train, y_train)
    print(f"\nTraining on {len(Xa)} rows "
          f"({'augmented x' + str(args.augment + 1) if args.augment else 'no augmentation'}) ...")
    random_model = build()
    random_model.fit(Xa, ya)
    random_accuracy = accuracy_score(y_test, random_model.predict(X_test))

    # ---------------------------------------------------------- block split
    train_parts, test_parts = [], []
    for frame in per_letter.values():
        cut = int(len(frame) * (1 - args.test_size))
        train_parts.append(frame.iloc[:cut])
        test_parts.append(frame.iloc[cut:])
    block_train = pd.concat(train_parts, ignore_index=True)
    block_test = pd.concat(test_parts, ignore_index=True)

    Xb, yb = prepare(
        block_train[feature_columns], block_train["label"]
    )
    block_model = build()
    block_model.fit(Xb, yb)
    block_predictions = block_model.predict(block_test[feature_columns])
    block_accuracy = accuracy_score(block_test["label"], block_predictions)

    print()
    print("=" * 58)
    print(f"  random-split accuracy : {random_accuracy * 100:6.2f}%   (flattering)")
    print(f"  block-split accuracy  : {block_accuracy * 100:6.2f}%   <- judge by this")
    print(f"  generalisation gap    : {(random_accuracy - block_accuracy) * 100:6.2f} points")
    print("=" * 58)
    print()
    print(classification_report(
        block_test["label"], block_predictions, zero_division=0
    ))

    # ------------------------------------------------- fit the shipped model
    # The model that gets saved is trained on everything available.
    Xf, yf = prepare(X, y)
    print(f"Fitting the final model on all {len(Xf)} rows ...")
    final_model = build()
    final_model.fit(Xf, yf)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.no_backup:
        backup = args.output.with_suffix(args.output.suffix + ".bak")
        shutil.copy2(args.output, backup)
        print(f"Previous model backed up to {backup}")

    joblib.dump(final_model, args.output)
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Model saved to {args.output} ({size_mb:.1f} MB)")
    if size_mb > 150:
        print("  note: that is a large file. Re-run with --min-samples-leaf 2 "
              "to shrink it at almost no cost in accuracy.")
    print("\nCheck it live before trusting it: python -m ml.scripts.live_demo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
