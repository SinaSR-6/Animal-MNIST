"""
transfer_learning.py
===============================================================
Run:  python 06_transfer_learning_v2.py --seeds 3 --epochs 15
Outputs: transfer_results_v2.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import common
import tensorflow as tf
from tensorflow.keras import layers, models


def backbone(input_shape=(28, 28, 1)):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    feat = layers.Dense(128, activation="relu", name="feat")(x)
    out = layers.Dense(10, activation="softmax")(feat)
    return models.Model(inp, out)


def prep(X):
    return X.reshape(-1, 28, 28, 1).astype("float32") / 255.0


def features(model, X):
    fm = models.Model(model.input, model.get_layer("feat").output)
    return fm.predict(prep(X), verbose=0)


def train_backbone(X, y, seed, epochs):
    tf.keras.utils.set_random_seed(seed)
    m = backbone()
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(prep(X), y, epochs=epochs, batch_size=128, validation_split=0.1, verbose=0,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,
                     restore_best_weights=True)])
    return m


def probe(bk, Xt, yt, seed):
    Xtr, Xte, ytr, yte = common.stratified_split(Xt, yt, 0.2, seed=seed)
    clf = LogisticRegression(max_iter=500).fit(features(bk, Xtr), ytr)
    return accuracy_score(yte, clf.predict(features(bk, Xte)))


def combine(d1, d2):
    return (np.concatenate([d1[0], d2[0]]), np.concatenate([d1[1], d2[1]]))


def run(seeds=3, epochs=15):
    data = common.load_all_three()
    if "MNIST" not in data:
        raise SystemExit("Needs keras (MNIST/Fashion).")
    mnist, fashion, animal = data["MNIST"], data["Fashion-MNIST"], data["Animal-MNIST"]
    src_mf = combine(mnist, fashion)

    # target -> (source_for_transfer, target_data)
    setups = {
        "-> Animal":  (src_mf, animal),
        "-> MNIST":   (animal, mnist),
        "-> Fashion": (animal, fashion),
    }
    rows = []
    for tag, (src, tgt) in setups.items():
        Xt, yt = tgt
        rnd, src_acc, same = [], [], []
        for s in range(seeds):
            tf.keras.utils.set_random_seed(s)
            rnd.append(probe(backbone(), Xt, yt, s))            # RANDOM features
            src_acc.append(probe(train_backbone(*src, s, epochs), Xt, yt, s))  # SOURCE
            same.append(probe(train_backbone(Xt, yt, s, epochs), Xt, yt, s))   # SAME
            tf.keras.backend.clear_session()
        for name, arr in [("RANDOM", rnd), ("SOURCE(transfer)", src_acc), ("SAME(ceiling)", same)]:
            arr = np.array(arr)
            rows.append({"target": tag, "backbone": name,
                         "mean_acc": arr.mean(), "std_acc": arr.std(ddof=1) if seeds>1 else 0.0})
            print(f"{tag:11s} | {name:17s} | {arr.mean():.4f} +/- "
                  f"{(arr.std(ddof=1) if seeds>1 else 0):.4f}")
        print("-" * 50)
    pd.DataFrame(rows).to_csv("transfer_results_v2.csv", index=False)
    print("Saved transfer_results_v2.csv")
    print("\nRead it as: if SOURCE ~ SAME >> RANDOM, transfer is real; "
          "if SOURCE ~ RANDOM, the task is just linearly easy.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    run(a.seeds, a.epochs)
