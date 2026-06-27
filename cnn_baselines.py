"""
cnn_baselines.py

Run:  python 02_cnn_baselines_fixed.py --seeds 3 --epochs 25
      python 02_cnn_baselines_fixed.py --seeds 3 --epochs 30 --with_resnet
Outputs: cnn_baselines_fixed.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from animal_mnist.dataset_Loader import (
    load_animal_mnist_dataset,
    load_mnist_data,
    load_fashion_mnist_data,
)


def stratified_split(X, y, test_size=0.2, seed=0):
    return train_test_split(X, y, test_size=test_size, random_state=seed,
                            shuffle=True, stratify=y)


def load_all_three():
    Xa, ya = load_animal_mnist_dataset()
    return {
        "Animal-MNIST": (np.asarray(Xa), np.asarray(ya).astype(int)),
        "MNIST": load_mnist_data(),
        "Fashion-MNIST": load_fashion_mnist_data(),
    }


def lenet(input_shape=(28, 28, 1), n=10):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(6, 5, padding="same", activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, 5, activation="relu"),
        layers.AveragePooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(120, activation="relu"),
        layers.Dense(84, activation="relu"),
        layers.Dense(n, activation="softmax"),
    ])
    m.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def small_cnn(input_shape=(28, 28, 1), n=10):
    m = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPool2D(pool_size=(2, 2)),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(), layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(n, activation="softmax"),
    ])
    m.compile(tf.keras.optimizers.Adam(1e-3),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def _res_block(x, f, stride=1):
    sc = x
    y = layers.Conv2D(f, 3, strides=stride, padding="same", use_bias=False)(x)
    y = layers.BatchNormalization()(y); y = layers.ReLU()(y)
    y = layers.Conv2D(f, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if stride != 1 or sc.shape[-1] != f:
        sc = layers.Conv2D(f, 1, strides=stride, use_bias=False)(sc)
        sc = layers.BatchNormalization()(sc)
    return layers.ReLU()(layers.Add()([sc, y]))


def resnet8(input_shape=(28, 28, 1), n=10):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = _res_block(x, 16); x = _res_block(x, 32, 2); x = _res_block(x, 64, 2)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(n, activation="softmax")(x)
    m = models.Model(inp, out)
    m.compile(tf.keras.optimizers.Adam(5e-4),
              "sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def prep(X):
    return X.reshape(-1, 28, 28, 1).astype("float32") / 255.0


def fit_eval(builder, X, y, seed, epochs):
    tf.keras.utils.set_random_seed(seed)
    Xtr, Xte, ytr, yte = stratified_split(X, y, 0.2, seed=seed)
    m = builder()
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
                                         patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=3, min_lr=1e-5),
    ]
    m.fit(prep(Xtr), ytr, epochs=epochs, batch_size=128, validation_split=0.1,
          verbose=0, callbacks=cbs)
    _, acc = m.evaluate(prep(Xte), yte, verbose=0)
    tf.keras.backend.clear_session()
    return acc


def run(seeds=3, epochs=25, with_resnet=False):
    builders = {"LeNet-5": lenet, "SmallCNN": small_cnn}
    if with_resnet:
        builders["ResNet-8"] = resnet8
    data = load_all_three()
    rows = []
    for mname, b in builders.items():
        for dname, (X, y) in data.items():
            accs = np.array([fit_eval(b, X, y, s, epochs) for s in range(seeds)])
            rows.append({"model": mname, "dataset": dname,
                         "mean_acc": accs.mean(), "std_acc": accs.std(),
                         "seeds": seeds})
            print(f"{mname:9s} | {dname:14s} | {accs.mean():.4f} +/- {accs.std():.4f}")
    pd.DataFrame(rows).to_csv("cnn_baselines_fixed.csv", index=False)
    print("\nSaved cnn_baselines_fixed.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--with_resnet", action="store_true")
    a = ap.parse_args()
    run(a.seeds, a.epochs, a.with_resnet)
