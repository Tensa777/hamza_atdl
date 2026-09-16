# ============================================================
# TASK 4
# FULL-PRECISION RESNET18 BASELINE
# ============================================================
#
# Important:
#   - Standard FP32 ResNet18
#   - CIFAR-10
#   - NO Knowledge Distillation
#   - NO ternary quantization
#   - Trained from scratch
#
# This model is used as the baseline for comparison with:
#
#   1. ResNet34 Teacher
#   2. Ternary ResNet18 Student
#
# ============================================================

import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Hamza_atdl\cifar-10-batches-py"

CHECKPOINT_PATH = (
    r"C:\Hamza_atdl\checkpoints"
    r"\resnet18_cifar10_baseline_best.pth"
)

EPOCHS = 100

BATCH_SIZE = 128

LEARNING_RATE = 0.1

MOMENTUM = 0.9

WEIGHT_DECAY = 5e-4

LABEL_SMOOTHING = 0.1

NUM_WORKERS = 0

SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# CIFAR DATASET
# ============================================================

class CIFAR10PythonDataset(Dataset):

    def __init__(
        self,
        images,
        labels,
        transform=None
    ):

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        image = self.images[index]

        label = int(
            self.labels[index]
        )

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# LOAD CIFAR BATCH
# ============================================================

def load_batch(path):

    with open(
        path,
        "rb"
    ) as f:

        data = pickle.load(
            f,
            encoding="bytes"
        )

    images = data[b"data"]

    labels = data[b"labels"]

    images = images.reshape(
        -1,
        3,
        32,
        32
    )

    images = np.transpose(
        images,
        (0, 2, 3, 1)
    )

    return (
        images,
        np.array(labels)
    )


# ============================================================
# LOAD CIFAR-10
# ============================================================

def load_cifar10():

    train_images = []
    train_labels = []

    for i in range(1, 6):

        path = os.path.join(
            DATASET_PATH,
            f"data_batch_{i}"
        )

        images, labels = load_batch(
            path
        )

        train_images.append(
            images
        )

        train_labels.append(
            labels
        )

    train_images = np.concatenate(
        train_images,
        axis=0
    )

    train_labels = np.concatenate(
        train_labels,
        axis=0
    )

    test_path = os.path.join(
        DATASET_PATH,
        "test_batch"
    )

    test_images, test_labels = load_batch(
        test_path
    )

    return (
        train_images,
        train_labels,
        test_images,
        test_labels
    )


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders():

    print("\nLoading CIFAR-10...")

    (
        train_images,
        train_labels,
        test_images,
        test_labels
    ) = load_cifar10()

    # --------------------------------------------------------
    # Same 45k / 5k split used previously
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(SEED)

    indices = torch.randperm(
        len(train_images),
        generator=generator
    ).numpy()

    train_indices = indices[:45000]

    val_indices = indices[45000:]

    train_x = train_images[
        train_indices
    ]

    train_y = train_labels[
        train_indices
    ]

    val_x = train_images[
        val_indices
    ]

    val_y = train_labels[
        val_indices
    ]

    # --------------------------------------------------------
    # CIFAR-10 normalization
    # --------------------------------------------------------

    mean = (
        0.4914,
        0.4822,
        0.4465
    )

    std = (
        0.2470,
        0.2435,
        0.2616
    )

    # --------------------------------------------------------
    # Training transform
    # --------------------------------------------------------

    train_transform = transforms.Compose([

        transforms.ToPILImage(),

        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean,
            std
        )
    ])

    # --------------------------------------------------------
    # Evaluation transform
    # --------------------------------------------------------

    eval_transform = transforms.Compose([

        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean,
            std
        )
    ])

    train_dataset = CIFAR10PythonDataset(
        train_x,
        train_y,
        train_transform
    )

    val_dataset = CIFAR10PythonDataset(
        val_x,
        val_y,
        eval_transform
    )

    test_dataset = CIFAR10PythonDataset(
        test_images,
        test_labels,
        eval_transform
    )

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

    print(
        f"Train: {len(train_dataset):,}"
    )

    print(
        f"Validation: {len(val_dataset):,}"
    )

    print(
        f"Test: {len(test_dataset):,}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# STANDARD CIFAR-10 RESNET18
# ============================================================

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_planes,
        planes,
        stride=1
    ):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            planes
        )

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(
            planes
        )

        self.shortcut = nn.Sequential()

        if (
            stride != 1
            or in_planes != planes
        ):

            self.shortcut = nn.Sequential(

                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm2d(
                    planes
                )
            )

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(
            out,
            inplace=True
        )

        out = self.conv2(out)

        out = self.bn2(out)

        out = out + identity

        out = F.relu(
            out,
            inplace=True
        )

        return out


# ============================================================
# RESNET18
# ============================================================

class ResNet18(nn.Module):

    def __init__(
        self,
        num_classes=10
    ):

        super().__init__()

        self.in_planes = 64

        # CIFAR-10 stem
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(
            64
        )

        self.layer1 = self._make_layer(
            64,
            2,
            stride=1
        )

        self.layer2 = self._make_layer(
            128,
            2,
            stride=2
        )

        self.layer3 = self._make_layer(
            256,
            2,
            stride=2
        )

        self.layer4 = self._make_layer(
            512,
            2,
            stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )

        self.fc = nn.Linear(
            512,
            num_classes
        )

        self._initialize_weights()

    # ========================================================
    # MAKE LAYER
    # ========================================================

    def _make_layer(
        self,
        planes,
        blocks,
        stride
    ):

        layers = []

        layers.append(
            BasicBlock(
                self.in_planes,
                planes,
                stride
            )
        )

        self.in_planes = planes

        for _ in range(
            1,
            blocks
        ):

            layers.append(
                BasicBlock(
                    self.in_planes,
                    planes,
                    1
                )
            )

        return nn.Sequential(
            *layers
        )

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(
                module,
                nn.BatchNorm2d
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.Linear
            ):

                nn.init.kaiming_uniform_(
                    module.weight,
                    a=5 ** 0.5
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        out = self.conv1(x)

        out = self.bn1(out)

        out = F.relu(
            out,
            inplace=True
        )

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.avgpool(out)

        out = torch.flatten(
            out,
            1
        )

        out = self.fc(out)

        return out


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    total_loss = 0.0

    correct = 0

    total = 0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        targets = targets.to(
            device,
            non_blocking=True
        )

        outputs = model(
            images
        )

        loss = F.cross_entropy(
            outputs,
            targets
        )

        batch_size = targets.size(0)

        total_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += batch_size

    return (
        total_loss / total,
        100.0 * correct / total
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_accuracy,
    train_accuracy
):

    os.makedirs(
        os.path.dirname(
            CHECKPOINT_PATH
        ),
        exist_ok=True
    )

    torch.save(
        {
            "epoch": epoch,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "val_accuracy":
                val_accuracy,

            "train_accuracy":
                train_accuracy
        },
        CHECKPOINT_PATH
    )


# ============================================================
# TRAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)

    print(
        "TASK 4: FULL-PRECISION RESNET18 BASELINE"
    )

    print("=" * 70)

    print(
        "\nTraining from scratch."
    )

    print(
        "Knowledge Distillation: DISABLED"
    )

    print(
        "Ternary Quantization: DISABLED"
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "CUDA:",
            torch.version.cuda
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = create_dataloaders()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet18(
        num_classes=10
    ).to(device)

    parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTrainable parameters: "
        f"{parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.SGD(

        model.parameters(),

        lr=LEARNING_RATE,

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY,

        nesterov=True
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS,

        eta_min=1e-4
    )

    # --------------------------------------------------------
    # Best values
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    best_epoch = 0

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        correct = 0

        total = 0

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        for batch_idx, (
            images,
            targets
        ) in enumerate(train_loader):

            images = images.to(
                device,
                non_blocking=True
            )

            targets = targets.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            outputs = model(
                images
            )

            loss = F.cross_entropy(
                outputs,
                targets,
                label_smoothing=LABEL_SMOOTHING
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            batch_size = targets.size(0)

            running_loss += (
                loss.item()
                *
                batch_size
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == targets
            ).sum().item()

            total += batch_size

            if (
                batch_idx == 0
                or
                (batch_idx + 1) % 100 == 0
                or
                batch_idx == len(train_loader) - 1
            ):

                current_accuracy = (
                    100.0
                    *
                    correct
                    /
                    total
                )

                print(
                    f"\rBatch "
                    f"{batch_idx + 1}/"
                    f"{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Accuracy: "
                    f"{current_accuracy:.2f}%",
                    end=""
                )

        print()

        train_loss = (
            running_loss
            /
            total
        )

        train_accuracy = (
            100.0
            *
            correct
            /
            total
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device
        )

        scheduler.step()

        # ----------------------------------------------------
        # Results
        # ----------------------------------------------------

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"Val Loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_accuracy,
                train_accuracy
            )

            print(
                "\n*** BEST BASELINE CHECKPOINT SAVED ***"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print(
        "\nLoading best baseline..."
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    # ========================================================
    # FINAL EVALUATION
    # ========================================================

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device
    )

    train_loss, final_train_accuracy = evaluate(
        model,
        train_loader,
        device
    )

    train_test_gap = (
        final_train_accuracy
        -
        test_accuracy
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TASK 4 BASELINE RESULTS"
    )

    print("=" * 70)

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Final Train Accuracy: "
        f"{final_train_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Train/Test Gap: "
        f"{train_test_gap:.2f} percentage points"
    )

    print(
        "\nCheckpoint:"
    )

    print(
        CHECKPOINT_PATH
    )

    print("=" * 70)

    print(
        "\nFull-precision ResNet18 baseline completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()