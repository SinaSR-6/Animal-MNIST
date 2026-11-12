import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


CLASS_NAMES = (
    "Bear", "Bird", "Cat", "Cow", "Dog", "Elephant",
    "Giraffe", "Horse", "Sheep", "Zebra",
)


def load_data(path=None):
    if path is not None:
        with np.load(path, allow_pickle=False) as data:
            image_key = "images" if "images" in data else "X"
            label_key = "labels" if "labels" in data else "y"
            images = data[image_key].copy()
            labels = data[label_key].copy()
            keys = ("train_idx", "val_idx", "test_idx")
            splits = tuple(data[k].copy() for k in keys) if all(k in data for k in keys) else None
        return images, labels, splits

    try:
        from animal_mnist.dataset_Loader import (
            load_animal_mnist_dataset,
            load_animal_mnist_splits,
        )
    except ImportError:
        from dataset_Loader import load_animal_mnist_dataset, load_animal_mnist_splits

    images, labels = load_animal_mnist_dataset()
    splits = load_animal_mnist_splits()
    return images, labels, splits


def fingerprint(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def audit_dataset(images, labels, splits=None, output_dir="integrity_report"):
    images = np.asarray(images)
    labels = np.asarray(labels)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    checks = {}
    checks["image_shape"] = images.shape == (10000, 28, 28)
    checks["image_dtype"] = images.dtype == np.uint8
    checks["label_shape"] = labels.shape == (10000,)
    checks["pixel_range"] = bool(images.size and np.issubdtype(images.dtype, np.number)
                                  and np.isfinite(images).all()
                                  and images.min() >= 0 and images.max() <= 255)

    valid_labels = (labels.ndim == 1 and np.issubdtype(labels.dtype, np.integer)
                    and bool(np.all((labels >= 0) & (labels < 10))))
    checks["label_values"] = bool(valid_labels)
    counts = np.bincount(labels.astype(np.int64), minlength=10)[:10] if valid_labels else np.zeros(10, dtype=int)
    checks["class_balance"] = bool(valid_labels and np.array_equal(counts, np.full(10, 1000)))

    split_counts = None
    split_hash = None
    if splits is not None:
        if len(splits) != 3:
            checks["split_structure"] = False
        else:
            raw_splits = [np.asarray(indices) for indices in splits]
            checks["split_structure"] = all(
                indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
                for indices in raw_splits
            )
            if checks["split_structure"]:
                train, val, test = (np.asarray(indices, dtype=np.int64) for indices in raw_splits)
                checks["split_sizes"] = tuple(map(len, (train, val, test))) == (7200, 800, 2000)
                combined = np.concatenate((train, val, test))
                checks["split_disjoint_complete"] = bool(
                    len(combined) == len(labels)
                    and np.array_equal(np.sort(combined), np.arange(len(labels)))
                )
                if checks["split_disjoint_complete"] and valid_labels:
                    split_counts = np.stack([
                        np.bincount(labels[indices].astype(np.int64), minlength=10)[:10]
                        for indices in (train, val, test)
                    ], axis=1)
                    checks["split_class_balance"] = bool(
                        np.array_equal(split_counts, np.tile([720, 80, 200], (10, 1)))
                    )
                else:
                    checks["split_class_balance"] = False
                split_hash = fingerprint(*raw_splits)
    else:
        checks["split_available"] = False

    with (output / "class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "total", "train", "validation", "test"])
        for i, name in enumerate(CLASS_NAMES):
            row = split_counts[i].tolist() if split_counts is not None else ["", "", ""]
            writer.writerow([i, name, int(counts[i]), *row])

    report = {
        "images": int(len(images)),
        "image_shape": list(images.shape),
        "image_dtype": str(images.dtype),
        "pixel_min": float(images.min()) if images.size else None,
        "pixel_max": float(images.max()) if images.size else None,
        "class_counts": {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)},
        "splits_checked": splits is not None,
        "checks": checks,
        "passed": all(checks.values()),
        "dataset_sha256": fingerprint(images, labels),
        "split_sha256": split_hash,
    }
    (output / "integrity_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"Dataset SHA-256: {report['dataset_sha256']}")
    if split_hash:
        print(f"Split SHA-256:   {split_hash}")
    print(f"Reports: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path)
    parser.add_argument("--output-dir", default="integrity_report")
    args = parser.parse_args()
    report = audit_dataset(*load_data(args.npz), output_dir=args.output_dir)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
