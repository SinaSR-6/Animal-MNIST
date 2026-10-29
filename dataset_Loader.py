from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import os
import gzip
import pickle
import importlib.resources as pkg_resources
from animal_mnist import data
import os
import gzip
import pickle
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

def get_data_path(filename):
    """Return the absolute path of a data file in the installed package."""
    package_dir = os.path.dirname(__file__)  # Get the installed package directory
    return os.path.join(package_dir, "data", "Animal_MNIST", filename)

def sample_data(data, labels, num_samples=10000):
    """Randomly sample a subset of data."""
    indices = np.random.choice(len(data), num_samples, replace=False)
    return data[indices], labels[indices]

def load_animal_mnist_dataset():
    """Load the Animal MNIST dataset from the installed package."""
    data_path = get_data_path("animal_data_version_5.gz")
    label_path = get_data_path("animal_label_version_5.gz")
    
    if not os.path.exists(data_path) or not os.path.exists(label_path):
        raise FileNotFoundError(f"Dataset files not found in package at: {data_path} and {label_path}")

    with gzip.open(data_path, 'rb') as f:
        data = pickle.load(f)
    with gzip.open(label_path, 'rb') as f:
        labels = pickle.load(f)

    return data, labels

def load_animal_mnist_splits():
    """
    Load the official Animal-MNIST v5 split indices. train_idx : ndarray of shape (7200,) | val_idx   : ndarray of shape (800,)| test_idx  : ndarray of shape (2000,)
    """
    split_path = get_data_path("animal_mnist_splits_v5.npz")

    with np.load(split_path, allow_pickle=False) as splits:
        train_idx = splits["train_idx"].copy()
        val_idx = splits["val_idx"].copy()
        test_idx = splits["test_idx"].copy()

    return train_idx, val_idx, test_idx

def load_mnist_data(num_samples=10000):
    """Load and sample MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_data, y_data = sample_data(np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test]), num_samples)
    return x_data, y_data

def load_fashion_mnist_data(num_samples=10000):
    """Load and sample Fashion-MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_data, y_data = sample_data(np.concatenate([x_train, x_test]), np.concatenate([y_train, y_test]), num_samples)
    return x_data, y_data
