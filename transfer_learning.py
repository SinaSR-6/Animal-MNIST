import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from experiment_data import NAMES, get_dataset, load_or_train, build_smallcnn, data_fingerprint

OUTPUT = Path("results/transfer")


def features(model, x):
    encoder = tf.keras.Model(model.inputs, model.get_layer("features").output)
    return encoder.predict(x, batch_size=256, verbose=0)


def train_target(name, seed, datasets, epochs):
    x, y, (tr, va, _) = datasets[name]
    path = OUTPUT / "models" / name / f"seed{seed}.weights.h5"
    return load_or_train(path, 10, x[tr], y[tr], x[va], y[va], seed, epochs)


def train_cross_source(seed, datasets, epochs):
    x1, y1, (tr1, va1, _) = datasets["MNIST"]
    x2, y2, (tr2, va2, _) = datasets["Fashion-MNIST"]
    path = OUTPUT / "models" / "MNIST_Fashion" / f"seed{seed}.weights.h5"
    return load_or_train(
        path, 20, np.concatenate((x1[tr1], x2[tr2])),
        np.concatenate((y1[tr1], y2[tr2] + 10)),
        np.concatenate((x1[va1], x2[va2])),
        np.concatenate((y1[va1], y2[va2] + 10)), seed, epochs
    )


def run(seeds, epochs):
    datasets = {name: get_dataset(name) for name in NAMES}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT / "per_seed.csv"
    rows = pd.read_csv(output_csv).to_dict("records") if output_csv.exists() else []
    for seed in seeds:
        rows = [row for row in rows if int(row["seed"]) != seed]
        targets = {name: train_target(name, seed, datasets, epochs) for name in NAMES}
        cross = train_cross_source(seed, datasets, epochs)
        for name in NAMES:
            x, y, (tr, va, te) = datasets[name]
            conditions = {
                "random": None,
                "cross_domain": cross if name == "Animal-MNIST" else targets["Animal-MNIST"],
                "target_trained": targets[name],
            }
            for condition, source in conditions.items():
                if source is None:
                    tf.keras.utils.set_random_seed(int(seed))
                    source = build_smallcnn()
                f_train = features(source, x[tr])
                f_val = features(source, x[va])
                f_test = features(source, x[te])
                scaler = StandardScaler().fit(f_train)
                f_train, f_val, f_test = [scaler.transform(a) for a in (f_train, f_val, f_test)]
                best = None
                for c in (0.1, 1.0, 10.0):
                    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=3000)
                    clf.fit(f_train, y[tr])
                    val_score = clf.score(f_val, y[va])
                    if best is None or val_score > best[0]:
                        best = (val_score, c, clf)
                val_score, c, clf = best
                score = clf.score(f_test, y[te])
                rows.append(dict(dataset=name, seed=seed, source=condition, selected_C=c,
                                 val_accuracy=val_score, test_accuracy=score,
                                 dataset_hash=data_fingerprint(x, y, (tr, va, te))))
                print(name, seed, condition, f"test={score:.4f} C={c}", flush=True)
                pd.DataFrame(rows).to_csv(output_csv, index=False)
        del targets, cross
        tf.keras.backend.clear_session()
    result = pd.DataFrame(rows)
    result.groupby(["dataset", "source"], sort=False).test_accuracy.agg(
        ["count", "mean", "std"]
    ).to_csv(OUTPUT / "summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run(args.seeds, args.epochs)
