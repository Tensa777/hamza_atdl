import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = r"C:\Hamza_atdl\cifar-10-batches-py"

BATCH_SIZE = 128
NUM_WORKERS = 0

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# LOAD CIFAR-10 BATCH
# ============================================================

def load_batch(file_path):

    with open(file_path, "rb") as f:
        batch = pickle.load(f, encoding="bytes")

    images = batch[b"data"]
    labels = batch[b"labels"]

    # Convert:
    # (N, 3072)
    #     -> (N, 3, 32, 32)
    #     -> (N, 32, 32, 3)

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, np.array(labels)


# ============================================================
# CUSTOM CIFAR-10 DATASET
# ============================================================

class CIFAR10Dataset(Dataset):

    def __init__(self, images, labels, transform=None):

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.images)

    def __getitem__(self, index):

        image = self.images[index]
        label = int(self.labels[index])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# LOAD ALL TRAINING DATA
# ============================================================

def load_training_data():

    images_list = []
    labels_list = []

    for i in range(1, 6):

        file_path = os.path.join(
            DATA_DIR,
            f"data_batch_{i}"
        )

        images, labels = load_batch(file_path)

        images_list.append(images)
        labels_list.append(labels)

        print(f"Loaded data_batch_{i}")

    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return images, labels


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    file_path = os.path.join(
        DATA_DIR,
        "test_batch"
    )

    return load_batch(file_path)


# ============================================================
# TRANSFORMS
# ============================================================

# Training:
# Random crop + horizontal flip
# This helps the ResNet34 generalize better.

train_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# Validation/Test:
# NO random augmentation.

test_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(
        CIFAR10_MEAN,
        CIFAR10_STD
    )
])


# ============================================================
# CREATE DATASETS + DATALOADERS
# ============================================================

def create_dataloaders():

    print("Loading CIFAR-10...")

    # --------------------------------------------------------
    # Load complete training data
    # --------------------------------------------------------

    all_train_images, all_train_labels = load_training_data()

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_images, test_labels = load_test_data()

    print(
        "\nTotal training images:",
        len(all_train_images)
    )

    print(
        "Total test images:",
        len(test_images)
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    train_size = 45000
    val_size = 5000

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(all_train_images),
        generator=generator
    )

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # --------------------------------------------------------
    # Separate arrays
    # --------------------------------------------------------

    train_images = all_train_images[
        train_indices.numpy()
    ]

    train_labels = all_train_labels[
        train_indices.numpy()
    ]

    val_images = all_train_images[
        val_indices.numpy()
    ]

    val_labels = all_train_labels[
        val_indices.numpy()
    ]

    # ========================================================
    # CREATE SEPARATE DATASET OBJECTS
    # ========================================================

    train_dataset = CIFAR10Dataset(
        train_images,
        train_labels,
        transform=train_transform
    )

    val_dataset = CIFAR10Dataset(
        val_images,
        val_labels,
        transform=test_transform
    )

    test_dataset = CIFAR10Dataset(
        test_images,
        test_labels,
        transform=test_transform
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    print("\n========== DATALOADERS ==========")

    print(
        "Training samples   :",
        len(train_dataset)
    )

    print(
        "Validation samples :",
        len(val_dataset)
    )

    print(
        "Test samples       :",
        len(test_dataset)
    )

    print(
        "Batch size         :",
        BATCH_SIZE
    )

    print(
        "Training batches   :",
        len(train_loader)
    )

    print(
        "Validation batches :",
        len(val_loader)
    )

    print(
        "Test batches       :",
        len(test_loader)
    )

    print("=================================")

    return train_loader, val_loader, test_loader


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_dataloaders()

    # --------------------------------------------------------
    # Check training batch
    # --------------------------------------------------------

    train_images, train_labels = next(
        iter(train_loader)
    )

    # --------------------------------------------------------
    # Check validation batch
    # --------------------------------------------------------

    val_images, val_labels = next(
        iter(val_loader)
    )

    # --------------------------------------------------------
    # Check test batch
    # --------------------------------------------------------

    test_images, test_labels = next(
        iter(test_loader)
    )

    print("\n========== SAMPLE BATCHES ==========")

    print(
        "Training images :",
        train_images.shape
    )

    print(
        "Training labels :",
        train_labels.shape
    )

    print(
        "Validation images:",
        val_images.shape
    )

    print(
        "Validation labels:",
        val_labels.shape
    )

    print(
        "Test images     :",
        test_images.shape
    )

    print(
        "Test labels     :",
        test_labels.shape
    )

    print(
        "Image dtype     :",
        train_images.dtype
    )

    print(
        "Label dtype     :",
        train_labels.dtype
    )

    print("====================================")