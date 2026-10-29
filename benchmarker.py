"""Reproducible Animal-MNIST v5 benchmark helper.

Public entry points retained:
    benchmark_model(model, X, y, model_name, dataset_name, repeats=5, is_tf_model=False)
    run_benchmark(models, repeats=5)
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from animal_mnist import dataset_Loader


DATASET_SIZE = 10_000
SAMPLE_SEED = 42
SPLIT_SEED = 42


def _check_split(y, split):
    """Validate disjoint indices, complete coverage, and intended class balance."""
    train_idx, val_idx, test_idx = (np.asarray(x, dtype=np.int64) for x in split)
    n = len(y)
    joined = np.concatenate([train_idx, val_idx, test_idx])
    if not np.array_equal(np.sort(joined), np.arange(n, dtype=np.int64)):
        raise ValueError('Train/validation/test indices must be disjoint and cover every image.')
    if n == DATASET_SIZE and len(np.unique(y)) == 10:
        if (len(train_idx), len(val_idx), len(test_idx)) != (7200, 800, 2000):
            raise ValueError('Expected 7200/800/2000 for the 10,000-image benchmark.')
        counts = {int(lab): (int(np.sum(y[train_idx] == lab)),
                             int(np.sum(y[val_idx] == lab)),
                             int(np.sum(y[test_idx] == lab)))
                  for lab in np.unique(y)}
        if any(counts[lab] != (720, 80, 200) for lab in counts):
            raise ValueError(f'Split is not class-balanced: {counts}')
    return train_idx, val_idx, test_idx


def _stratified_split(y, seed=SPLIT_SEED):
    """Exactly 20% test, then 10% of the remaining 80% for validation."""
    ids = np.arange(len(y), dtype=np.int64)
    train_val, test = train_test_split(ids, test_size=0.20,
                                      random_state=seed, stratify=y)
    train, val = train_test_split(train_val, test_size=0.10,
                                  random_state=seed, stratify=y[train_val])
    return _check_split(y, (np.sort(train), np.sort(val), np.sort(test)))


def _reference_data(which, seed=SAMPLE_SEED):
    """Create reproducible 10k balanced subsets of MNIST/Fashion-MNIST. Do NOT equate these with the official MNIST test splits.
    """
    if which == 'MNIST':
        load = tf.keras.datasets.mnist.load_data
    elif which == 'Fashion-MNIST':
        load = tf.keras.datasets.fashion_mnist.load_data
    else:
        raise ValueError(which)
    (x1, y1), (x2, y2) = load()
    X = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)
    rng = np.random.default_rng(seed)
    groups = []
    for lab in range(10):
        candidates = np.flatnonzero(y == lab)
        if len(candidates) < 1000:
            raise ValueError(f'Insufficient examples of class {lab} in {which}')
        groups.append(rng.choice(candidates, size=1000, replace=False))
    selection = np.concatenate(groups)
    rng.shuffle(selection)
    return X[selection], y[selection]


def _is_tf_model(model):
    return isinstance(model, tf.keras.Model)


def _new_model(template, is_tf_model, seed):
    """Fresh fit for EVERY dataset and run. Never reuses trained weights."""
    if is_tf_model:
        tf.keras.utils.set_random_seed(seed)
        try:
            new = tf.keras.models.clone_model(template)
            config = template.get_compile_config()
            if not config:
                raise ValueError('Keras model must be compiled before benchmarking.')
            new.compile_from_config(config)
        except Exception as exc:
            raise ValueError(
                'Cannot clone/compile the supplied Keras model. Use a built, '
                'compiled Sequential or Functional model with serializable '
                'components; or adapt this helper for your custom model.'
            ) from exc
        return new
    new = clone(template)
    params = new.get_params(deep=False)
    if 'random_state' in params:
        new.set_params(random_state=seed)
    return new


def _features(X, is_tf_model):
    X = np.asarray(X)
    if is_tf_model:
        # Existing package benchmark supported 28x28 grayscale CNNs.
        if X.ndim == 3 and X.shape[1:] == (28, 28):
            return X[..., np.newaxis].astype(np.float32) / 255.0
        if X.ndim == 4 and X.shape[1:] == (28, 28, 1):
            return X.astype(np.float32) / 255.0
        raise ValueError(f'Keras benchmark expects 28x28 grayscale images, got {X.shape}')
    if X.ndim > 2:
        return X.reshape(len(X), -1)
    return X


def benchmark_model(model, X, y, model_name, dataset_name, repeats=5,
                    is_tf_model=False, *, split_indices=None, seeds=None,
                    epochs=1, batch_size=32, patience=5, split_seed=SPLIT_SEED):
    """Evaluate one model with a fixed, stratified train/val/test partition.

    repeats: independently initialized fits, NOT repeated data splits.
    epochs: Keras only (default 1 for backward compatibility). For CNN research,
            set explicitly, e.g. epochs=30, and document the training protocol.
    seeds: e.g. [0,1,2,3,4]; overrides repeats when supplied.
    """
    X, y = np.asarray(X), np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError('Image and label counts do not match.')
    if split_indices is None:
        if dataset_name in {'Animal-MNIST Dataset', 'Animal-MNIST'}:
            split_indices = dataset_Loader.load_animal_mnist_splits()
        else:
            split_indices = _stratified_split(y, seed=split_seed)
    train_idx, val_idx, test_idx = _check_split(y, split_indices)

    seeds = list(range(repeats)) if seeds is None else list(seeds)
    if not seeds or any(not isinstance(s, (int, np.integer)) for s in seeds):
        raise ValueError('Supply one or more integer training seeds.')
    if epochs < 1 or batch_size < 1 or patience < 0:
        raise ValueError('epochs/batch_size must be positive; patience >= 0.')

    X_train, X_val, X_test = (_features(X[idx], is_tf_model) for idx in
                              (train_idx, val_idx, test_idx))
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    accs, val_accs, times = [], [], []
    for seed in seeds:
        candidate = _new_model(model, is_tf_model, int(seed))
        start = time.perf_counter()
        if is_tf_model:
            callbacks = [tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience,
                restore_best_weights=True)] if epochs > 1 else []
            candidate.fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=epochs, batch_size=batch_size,
                          verbose=0, callbacks=callbacks)
        else:
            # Fixed hyperparameters: validation is held aside, not used for fitting.
            candidate.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        if is_tf_model:
            val_pred = np.argmax(candidate.predict(X_val, verbose=0), axis=-1)
            test_pred = np.argmax(candidate.predict(X_test, verbose=0), axis=-1)
            # Do not keep all repeated CNN instances and their GPU memory.
            del candidate
            tf.keras.backend.clear_session()
        else:
            val_pred = candidate.predict(X_val)
            test_pred = candidate.predict(X_test)
        val_accs.append(float(accuracy_score(y_val, val_pred)))
        accs.append(float(accuracy_score(y_test, test_pred)))
        times.append(float(elapsed))
        print(f'  {model_name} / {dataset_name} / seed {int(seed)}: '
              f'val={val_accs[-1]:.4f}, test={accs[-1]:.4f}, fit={elapsed:.2f}s',
              flush=True)
    return {
        'Model': model_name,
        'Dataset': dataset_name,
        'Accuracy (mean)': float(np.mean(accs)),
        'Accuracy (std)': float(np.std(accs, ddof=1)) if len(accs)>1 else 0.0,
        'Validation Accuracy (mean)': float(np.mean(val_accs)),
        'Validation Accuracy (std)': float(np.std(val_accs, ddof=1)) if len(val_accs)>1 else 0.0,
        'Training Time (mean)': float(np.mean(times)),
        'Training Time (std)': float(np.std(times, ddof=1)) if len(times)>1 else 0.0,
        'Runs': len(seeds),
        'Seeds': ','.join(str(s) for s in seeds),
        'Split': 'official-v5' if dataset_name in {'Animal-MNIST Dataset', 'Animal-MNIST'} else f'stratified-{split_seed}',
    }


def run_benchmark(models, repeats=5, *, seeds=None, epochs=1, batch_size=32,
                  patience=5, sample_seed=SAMPLE_SEED, split_seed=SPLIT_SEED):
    """Run three-dataset comparisons with consistent 72/8/20 evaluation sizes.

    Keeps the original run_benchmark(models, repeats=5) call working.
    For Keras models, explicitly set epochs to match the paper's protocol.
    Returned mean/std across seeds measures training variation on a FIXED split.
    """
    datasets = {
        'MNIST': _reference_data('MNIST', seed=sample_seed),
        'Fashion-MNIST': _reference_data('Fashion-MNIST', seed=sample_seed),
        'Animal-MNIST Dataset': dataset_Loader.load_animal_mnist_dataset(),
    }
    results = []
    for model_name, model in models.items():
        is_tf = _is_tf_model(model)
        for dataset_name, (X, y) in datasets.items():
            print(f'Running {model_name} on {dataset_name} using fixed 72/8/20...', flush=True)
            result = benchmark_model(
                model, X, y, model_name, dataset_name, repeats, is_tf,
                seeds=seeds, epochs=epochs, batch_size=batch_size,
                patience=patience, split_seed=split_seed)
            results.append(result)
    return pd.DataFrame(results)
