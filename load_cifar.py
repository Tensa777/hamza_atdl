import os
import pickle
import numpy as np

# CIFAR-10 dataset path
DATA_DIR = r"C:\Hamza_atdl\cifar-10-batches-py"


def load_batch(file_path):
    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    # CIFAR-10 format:
    # (10000, 3072) -> (10000, 3, 32, 32) -> (10000, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.array(labels)


# Load all 5 training batches
train_images = []
train_labels = []

for i in range(1, 6):
    file_path = os.path.join(DATA_DIR, f"data_batch_{i}")

    images, labels = load_batch(file_path)

    train_images.append(images)
    train_labels.append(labels)

    print(f"Loaded data_batch_{i}: {images.shape}")


# Combine training batches
x_train = np.concatenate(train_images, axis=0)
y_train = np.concatenate(train_labels, axis=0)


# Load test data
test_path = os.path.join(DATA_DIR, "test_batch")
x_test, y_test = load_batch(test_path)


# Load class names
meta_path = os.path.join(DATA_DIR, "batches.meta")

with open(meta_path, "rb") as f:
    meta = pickle.load(f, encoding="bytes")

class_names = [name.decode("utf-8") for name in meta[b"label_names"]]


print("\n========== CIFAR-10 DATASET ==========")
print("Training images :", x_train.shape)
print("Training labels :", y_train.shape)
print("Test images     :", x_test.shape)
print("Test labels     :", y_test.shape)
print("Number of classes:", len(class_names))
print("Classes          :", class_names)
print("======================================")