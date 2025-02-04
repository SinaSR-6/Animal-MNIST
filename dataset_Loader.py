
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np
import os
import gzip
import pickle


def sample_data(data, labels, num_samples=10000):
    indices = np.random.choice(len(data), num_samples, replace=False)
    return data[indices], labels[indices]


def load_animal_shape(data_path, label_path):
    with gzip.GzipFile(data_path, 'rb') as f:
        data = pickle.load(f)
    with gzip.GzipFile(label_path, 'rb') as f:
        label = pickle.load(f)
    return data, label


def load_mnist_data(num_samples=10000):
    """
    Load and sample MNIST dataset.
    
    Parameters:
    - num_samples (int): Number of samples to extract.

    Returns:
    - x_mnist (ndarray): Sampled MNIST images.
    - y_mnist (ndarray): Sampled MNIST labels.
    """
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
    x_mnist, y_mnist = sample_data(
        np.concatenate([x_train_mnist, x_test_mnist]),
        np.concatenate([y_train_mnist, y_test_mnist]),
        num_samples=num_samples
    )
    return x_mnist, y_mnist


def load_fashion_mnist_data(num_samples=10000):
    """
    Load and sample Fashion-MNIST dataset.
    
    Parameters:
    - num_samples (int): Number of samples to extract.

    Returns:
    - x_fashion (ndarray): Sampled Fashion-MNIST images.
    - y_fashion (ndarray): Sampled Fashion-MNIST labels.
    """
    (x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
    x_fashion, y_fashion = sample_data(
        np.concatenate([x_train_fashion, x_test_fashion]),
        np.concatenate([y_train_fashion, y_test_fashion]),
        num_samples=num_samples
    )
    return x_fashion, y_fashion


def locate_animal_files(expected_files):
    """
    Locate required animal dataset files, prompting the user if files are missing.

    Parameters:
    - expected_files (list): List of expected file paths.

    Returns:
    - file_paths (dict): Dictionary of located file paths with filenames as keys.
    """
    file_paths = {}
    for expected_file in expected_files:
        while True:
            if os.path.exists(expected_file):
                print(f"Data file found at: {expected_file}\n\n")
                file_paths[os.path.basename(expected_file)] = expected_file
                break
            else:
                print(f"{os.path.basename(expected_file)} not found!!\n")
                file_path = input(f'Please input the location of "{os.path.basename(expected_file)}": ')
                if os.path.isfile(file_path) and os.path.basename(file_path) == os.path.basename(expected_file):
                    print(f"Data file found at: {file_path}\n\n")
                    file_paths[os.path.basename(expected_file)] = file_path
                    break
                else:
                    print(f"Error: The file at {file_path} does not match '{os.path.basename(expected_file)}'. Please check the path and try again.\n")
    return file_paths


def load_animal_shape_dataset():
    """
    Locate animal dataset files, load the dataset, and return the data and labels.

    Returns:
    - X_custom (ndarray): Loaded animal dataset images.
    - y_custom (ndarray): Loaded animal dataset labels.
    """
    expected_files = [
        os.path.join("data", "Animal Shape", "animal_data_version_3.gz"),
        os.path.join("data", "Animal Shape", "animal_label_version_3.gz")
    ]
    file_paths = locate_animal_files(expected_files)
    DATA_PATH = file_paths["animal_data_version_3.gz"]
    LABEL_PATH = file_paths["animal_label_version_3.gz"]

    # Load the animal dataset
    X_custom, y_custom = load_animal_shape(DATA_PATH, LABEL_PATH)
    return X_custom, y_custom
