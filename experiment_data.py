import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

NAMES = ("MNIST", "Fashion-MNIST", "Animal-MNIST")


def get_dataset(name):
    if name == "Animal-MNIST":
        try:
            from animal_mnist.dataset_Loader import load_animal_mnist_dataset, load_animal_mnist_splits
        except ImportError:
            from dataset_Loader import load_animal_mnist_dataset, load_animal_mnist_splits
        x, y = load_animal_mnist_dataset()
        train, val, test = load_animal_mnist_splits()
    elif name in ("MNIST", "Fashion-MNIST"):
        from tensorflow.keras.datasets import mnist, fashion_mnist
        loader = mnist if name == "MNIST" else fashion_mnist
        (x_train, y_train), (x_test, y_test) = loader.load_data()
        x = np.concatenate((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        rng = np.random.default_rng(42)
        sample = np.concatenate([
            rng.choice(np.flatnonzero(y == label), size=1000, replace=False)
            for label in range(10)
        ])
        rng.shuffle(sample)
        x, y = x[sample], y[sample]
        indices = np.arange(len(y))
        pool, test = train_test_split(indices, test_size=0.2, stratify=y, random_state=42)
        train, val = train_test_split(pool, test_size=0.1, stratify=y[pool], random_state=42)
        train, val, test = np.sort(train), np.sort(val), np.sort(test)
    else:
        raise ValueError(name)

    x = np.asarray(x).reshape(-1, 28, 28).astype(np.float32)
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    if x.max() > 1.01:
        x /= 255.0
    x = np.clip(x, 0, 1)
    train, val, test = (np.asarray(i, dtype=np.int64) for i in (train, val, test))
    if x.shape != (10000, 28, 28) or not np.array_equal(np.bincount(y, minlength=10), np.full(10, 1000)):
        raise ValueError(f"Unexpected {name} data shape or class counts")
    if tuple(map(len, (train, val, test))) != (7200, 800, 2000):
        raise ValueError("Unexpected split sizes")
    if not np.array_equal(np.sort(np.concatenate((train, val, test))), np.arange(10000)):
        raise ValueError("Overlapping or missing split indices")
    for label in range(10):
        if tuple(np.sum(y[ids] == label) for ids in (train, val, test)) != (720, 80, 200):
            raise ValueError(f"Incorrect stratification for {name} class {label}")
    return x[..., None], y, (train, val, test)


def data_fingerprint(x, y, splits):
    h = hashlib.sha256()
    for a in (x, y, *splits):
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build_smallcnn(classes=10):
    import tensorflow as tf
    L = tf.keras.layers
    inputs = tf.keras.Input((28, 28, 1))
    x = inputs
    for units in (32, 64, 128):
        x = L.Conv2D(units, 3, padding="valid", activation="relu")(x)
        x = L.BatchNormalization(momentum=0.9)(x)
        x = L.MaxPooling2D()(x)
        x = L.Dropout(0.25)(x)
    x = L.Flatten()(x)
    x = L.Dense(128, activation="relu")(x)
    x = L.Dropout(0.5)(x)
    features = L.Dense(64, activation="relu", name="features")(x)
    output = L.Dense(classes, activation="softmax")(features)
    return tf.keras.Model(inputs, output)


def fit_smallcnn(model, train, labels, val, val_labels, epochs=100, batch=128):
    import tensorflow as tf
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, start_from_epoch=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=5, factor=0.5, min_lr=1e-5
        ),
    ]
    return model.fit(
        train, labels, validation_data=(val, val_labels), epochs=epochs,
        batch_size=batch, callbacks=callbacks, verbose=2
    )


def load_or_train(path, classes, train, labels, val, val_labels, seed, epochs=100):
    import tensorflow as tf
    path = Path(path)
    tf.keras.utils.set_random_seed(int(seed))
    model = build_smallcnn(classes)
    report = path.with_suffix(".json")
    digest = data_fingerprint(train, labels, (val, val_labels))
    metadata = dict(seed=int(seed), classes=int(classes), epochs=int(epochs), data_hash=digest)
    if path.exists():
        if not report.exists() or json.loads(report.read_text()) != metadata:
            raise ValueError(f"Checkpoint configuration mismatch: {path}")
        model.load_weights(str(path))
    else:
        if report.exists():
            raise ValueError(f"Checkpoint missing but metadata exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        fit_smallcnn(model, train, labels, val, val_labels, epochs=epochs)
        model.save_weights(str(path))
        report.write_text(json.dumps(metadata, indent=2))
    return model
