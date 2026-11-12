import argparse
import hashlib
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, binary_fill_holes, label
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset

try:
    from animal_mnist.dataset_Loader import load_animal_mnist_dataset
except ImportError:
    from dataset_Loader import load_animal_mnist_dataset

CLASSES = ("Bear", "Bird", "Cat", "Cow", "Dog", "Elephant", "Giraffe", "Horse", "Sheep", "Zebra")
REPRESENTATIONS = ("grayscale", "filled_silhouette", "outer_contour")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def representations(images):
    x = np.asarray(images).reshape(-1, 28, 28).astype(np.float32)
    if x.max() > 1.01:
        x /= 255.0
    x = np.clip(x, 0, 1)
    mask = np.stack([binary_fill_holes(img >= 0.25) for img in x]).astype(np.float32)
    contour = np.stack([
        (m.astype(bool) & ~binary_erosion(m.astype(bool), structure=np.ones((3, 3), bool)))
        for m in mask
    ]).astype(np.float32)
    return dict(zip(REPRESENTATIONS, (x, mask, contour)))


def make_folds(y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    for fold, (pool, test) in enumerate(cv.split(np.zeros(len(y)), y)):
        train, val = train_test_split(pool, test_size=0.1, stratify=y[pool], random_state=42+fold)
        if tuple(map(len, (train, val, test))) != (7200, 800, 2000):
            raise ValueError("Incorrect fold sizes")
        for idx, n in ((train, 720), (val, 80), (test, 200)):
            if not np.array_equal(np.bincount(y[idx], minlength=10), np.full(10, n)):
                raise ValueError("Incorrect class balance")
        folds.append((train, val, test))
    if not np.array_equal(np.sort(np.concatenate([s[2] for s in folds])), np.arange(len(y))):
        raise ValueError("Overlapping fold test sets")
    return folds


class AnimalData(Dataset):
    def __init__(self, x, y, indices):
        self.x, self.y, self.ids = x, y, np.asarray(indices)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        i = int(self.ids[index])
        return torch.from_numpy(self.x[i][None].copy()), int(self.y[i]), i


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)


def loader(x, y, ids, shuffle=False, seed=0, batch=128):
    return DataLoader(AnimalData(x, y, ids), batch_size=batch, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed),
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    rows = []
    losses = 0.0
    for x, y, ids in batches:
        x, y = x.to(DEVICE), y.to(DEVICE)
        scores = model(x)
        prediction = scores.argmax(1)
        losses += F.cross_entropy(scores, y, reduction="sum").item()
        rows.extend(zip(ids.numpy().tolist(), y.cpu().numpy().tolist(),
                        prediction.cpu().numpy().tolist()))
    df = pd.DataFrame(rows, columns=("image_id", "true", "prediction"))
    return df, losses / len(df)


def fit_one(fold, representation, x, y, indices, folder, epochs, batch):
    train, val, test = indices
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / "test_predictions.csv"
    summary = folder / "summary.json"
    dataset_hash = sha(np.asarray(x, dtype=np.float32))
    if out.exists() and summary.exists():
        previous = json.loads(summary.read_text())
        predictions = pd.read_csv(out)
        if previous["input_hash"] != dataset_hash or not np.array_equal(
                np.sort(predictions.image_id.values), np.sort(test)):
            raise ValueError(f"Stored results differ from input: {folder}")
        # Results written before these keys existed cannot be checked and are accepted as-is.
        for key, value in (("epochs_requested", epochs), ("batch", batch)):
            if key in previous and previous[key] != value:
                raise ValueError(
                    f"Stored results in {folder} used {key}={previous[key]}, "
                    f"but {key}={value} was requested. Delete the folder or use a new --out."
                )
        print(f"Reused {representation} fold {fold}", flush=True)
        return

    set_seed(1000 + fold)
    model = CNN().to(DEVICE)
    train_loader = loader(x, y, train, True, 1000+fold, batch)
    val_loader = loader(x, y, val, batch=batch)
    test_loader = loader(x, y, test, batch=batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    best_accuracy, best_loss, stale = -1.0, float("inf"), 0
    best_state = None
    history = []
    for epoch in range(1, epochs+1):
        model.train()
        correct = total = 0
        for images, labels, _ in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
                scores = model(images)
                loss = F.cross_entropy(scores, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += int((scores.argmax(1) == labels).sum())
            total += len(labels)
        val_pred, val_loss = evaluate(model, val_loader)
        val_accuracy = float((val_pred.true == val_pred.prediction).mean())
        history.append((epoch, correct / total, val_accuracy, val_loss))
        improved = val_accuracy > best_accuracy + 1e-10 or (
            abs(val_accuracy - best_accuracy) <= 1e-10 and val_loss < best_loss - 1e-8
        )
        if improved:
            best_accuracy, best_loss, stale = val_accuracy, val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 7:
            break
    torch.save(best_state, folder / "best.pt")
    model.load_state_dict(best_state)
    predictions, _ = evaluate(model, test_loader)
    if not np.array_equal(predictions.image_id.values, test):
        raise ValueError("Unexpected test image order")
    predictions.to_csv(out, index=False)
    pd.DataFrame(history, columns=("epoch", "train_accuracy", "val_accuracy", "val_loss")).to_csv(
        folder / "history.csv", index=False
    )
    summary.write_text(json.dumps(dict(
        fold=fold, representation=representation, input_hash=dataset_hash,
        test_accuracy=float((predictions.true == predictions.prediction).mean()),
        best_val_accuracy=best_accuracy, epochs=len(history),
        epochs_requested=epochs, batch=batch), indent=2))
    print(f"Completed {representation} fold {fold}: {len(predictions)} images", flush=True)


def summarize(folder, x, y):
    frames = []
    for fold in range(5):
        for representation in REPRESENTATIONS:
            file = folder / f"fold{fold}_{representation}" / "test_predictions.csv"
            if not file.exists():
                raise FileNotFoundError(file)
            data = pd.read_csv(file)
            data["fold"] = fold
            data["representation"] = representation
            frames.append(data)
    all_predictions = pd.concat(frames, ignore_index=True)
    overall = []
    per_class = []
    for representation in REPRESENTATIONS:
        part = all_predictions[all_predictions.representation == representation]
        if len(part) != len(y) or not np.array_equal(np.sort(part.image_id.values), np.arange(len(y))):
            raise ValueError("Incomplete out-of-fold predictions")
        part = part.sort_values("image_id")
        if not np.array_equal(part.true.values, y):
            raise ValueError("Out-of-fold labels do not match input")
        cm = confusion_matrix(part.true, part.prediction, labels=range(10))
        np.save(folder / f"confusion_{representation}.npy", cm)
        for label_id, label_name in enumerate(CLASSES):
            per_class.append(dict(representation=representation, class_name=label_name,
                                  correct=int(cm[label_id, label_id]),
                                  count=int(cm[label_id].sum()),
                                  accuracy=float(cm[label_id, label_id] / cm[label_id].sum())))
        horse, zebra = part[part.true == 7], part[part.true == 9]
        overall.append(dict(representation=representation, accuracy=np.trace(cm)/cm.sum(),
                            horse_zebra_accuracy=np.mean(part[part.true.isin((7, 9))].true ==
                                                         part[part.true.isin((7, 9))].prediction),
                            horse_to_zebra=int((horse.prediction == 9).sum()),
                            zebra_to_horse=int((zebra.prediction == 7).sum()),
                            direct_confusions=int((horse.prediction == 9).sum() + (zebra.prediction == 7).sum())))
    all_predictions.to_csv(folder / "all_oof_predictions.csv", index=False)
    pd.DataFrame(overall).to_csv(folder / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(folder / "per_class_accuracy.csv", index=False)
    mask = x["filled_silhouette"]
    components = [label(image, structure=np.ones((3, 3)))[1] for image in mask.astype(bool)]
    pd.DataFrame({"image_id": np.arange(len(y)), "class_id": y,
                  "mask_components": components}).to_csv(folder / "mask_quality.csv", index=False)
    print(pd.DataFrame(overall).to_string(index=False))
    print("Multiple-component masks:", sum(np.asarray(components) > 1))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, row in zip(axes, overall):
        cm = np.load(folder / f"confusion_{row['representation']}.npy")
        ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{row['representation']}: {row['accuracy']:.2%}")
        ax.set_xticks(range(10), CLASSES, rotation=90)
        ax.set_yticks(range(10), CLASSES)
    fig.tight_layout()
    fig.savefig(folder / "confusions.png", dpi=220)
    plt.close(fig)


def main(args):
    images, y = load_animal_mnist_dataset()
    y = np.asarray(y, dtype=np.int64).ravel()
    if len(y) != 10000 or not np.array_equal(np.bincount(y), np.full(10, 1000)):
        raise ValueError("Expected 10,000 balanced Animal-MNIST images")
    x = representations(images)
    area = float(x["filled_silhouette"].mean())
    if not 0.05 <= area <= 0.70:
        raise ValueError(
            f"Mean filled-mask area is {area:.2f}; expected bright animals on a dark "
            "background (check that the loader does not return black-on-white images)."
        )
    folds = make_folds(y)
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    for fold, indices in enumerate(folds):
        np.savez_compressed(folder / f"fold{fold}_split.npz", train=indices[0],
                            val=indices[1], test=indices[2])
        for representation in REPRESENTATIONS:
            fit_one(fold, representation, x[representation], y, indices,
                    folder / f"fold{fold}_{representation}", args.epochs, args.batch)
    summarize(folder, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/geometric_representation")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch", type=int, default=128)
    main(parser.parse_args())
