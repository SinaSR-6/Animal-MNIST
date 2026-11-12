import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from experiment_data import NAMES, get_dataset, load_or_train

OUTPUT = Path("results/joint_classification")


def run(seeds, epochs):
    datasets = {name: get_dataset(name) for name in NAMES}
    data = {}
    for split_index, part in enumerate(("train", "val", "test")):
        xs, ys, domains, ids = [], [], [], []
        for j, name in enumerate(NAMES):
            x, y, splits = datasets[name]
            selected = splits[split_index]
            xs.append(x[selected])
            ys.append(y[selected] + 10*j)
            domains.append(np.full(len(selected), j, dtype=np.int64))
            ids.append(selected)
        data[part] = (np.concatenate(xs), np.concatenate(ys),
                      np.concatenate(domains), np.concatenate(ids))
    assert [len(data[s][1]) for s in ("train", "val", "test")] == [21600, 2400, 6000]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT / "per_seed.csv"
    records = pd.read_csv(output_csv).to_dict("records") if output_csv.exists() else []
    for seed in seeds:
        records = [row for row in records if int(row["seed"]) != seed]
        xt, yt, _, _ = data["train"]
        xv, yv, _, _ = data["val"]
        model = load_or_train(
            OUTPUT / "models" / f"seed{seed}.weights.h5", 30, xt, yt, xv, yv, seed, epochs
        )
        x, y, domains, ids = data["test"]
        probabilities = model.predict(x, batch_size=128, verbose=0)
        pred = probabilities.argmax(1)
        rows = []
        for j, name in enumerate(NAMES):
            mask = domains == j
            true = y[mask]
            predicted = pred[mask]
            domain_acc = np.mean((predicted // 10) == j)
            acc = np.mean(predicted == true)
            restricted = probabilities[mask, 10*j:10*(j+1)].argmax(1) + 10*j
            rows.append(dict(seed=seed, dataset=name, accuracy_30class=acc,
                             accuracy_known_domain=np.mean(restricted == true),
                             domain_accuracy=domain_acc, n=len(true)))
        records.extend(rows)
        predictions = pd.DataFrame({
            "seed": seed, "dataset": np.asarray(NAMES)[domains], "image_id": ids,
            "true_label": y, "predicted_label": pred,
            "correct": (pred == y).astype(int)
        })
        predictions.to_csv(OUTPUT / f"predictions_seed{seed}.csv", index=False)
        pd.DataFrame(records).to_csv(output_csv, index=False)
        print(seed, rows, flush=True)
        del model
        tf.keras.backend.clear_session()
    df = pd.DataFrame(records)
    df.groupby("dataset", sort=False).accuracy_30class.agg(
        ["count", "mean", "std"]
    ).to_csv(OUTPUT / "summary.csv")
    print(f"30-class accuracy: {(df.accuracy_30class.mean()):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(args.seeds, args.epochs)
