# ============================================================
# TASK 3
# TERNARY RESNET18 + KNOWLEDGE DISTILLATION + QAT
# CIFAR-10
# ============================================================
#
# Teacher:
#   ResNet34 trained from scratch
#
# Student:
#   Ternary ResNet18
#
# Training:
#   Knowledge Distillation
#   +
#   Quantization-Aware Training
#
# QAT is active during EVERY forward pass because
# ternary_resnet18.py replaces Conv/FC weights with:
#
#       {-alpha, 0, +alpha}
#
# during the forward pass.
#
# ============================================================

import os
import pickle
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from ternary_resnet18 import TernaryResNet18
from resnet34_cifar import ResNet34


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = r"C:\Hamza_atdl\cifar-10-batches-py"

TEACHER_CHECKPOINT = (
    r"C:\Hamza_atdl\checkpoints"
    r"\resnet34_cifar10_teacher_best.pth"
)

STUDENT_CHECKPOINT = (
    r"C:\Hamza_atdl\checkpoints"
    r"\ternary_resnet18_kd_best.pth"
)

EPOCHS = 150

BATCH_SIZE = 128

LEARNING_RATE = 0.01

WEIGHT_DECAY = 5e-4

MOMENTUM = 0.9

# ------------------------------------------------------------
# Knowledge Distillation
# ------------------------------------------------------------

KD_WEIGHT = 0.4

CE_WEIGHT = 0.6

TEMPERATURE = 4.0

# ------------------------------------------------------------
# Label smoothing
# ------------------------------------------------------------

LABEL_SMOOTHING = 0.05

# ------------------------------------------------------------
# Gradient clipping
# ------------------------------------------------------------

GRAD_CLIP = 5.0

# ------------------------------------------------------------
# Workers
# ------------------------------------------------------------

NUM_WORKERS = 0

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# CIFAR-10 PYTHON DATASET
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
# LOAD CIFAR-10 PYTHON BATCH
# ============================================================

def load_batch(file_path):

    with open(
        file_path,
        "rb"
    ) as f:

        data = pickle.load(
            f,
            encoding="bytes"
        )

    images = data[b"data"]

    labels = data[b"labels"]

    # --------------------------------------------------------
    # CIFAR data is stored as:
    #
    # [N, 3072]
    #
    # 1024 red
    # 1024 green
    # 1024 blue
    #
    # Convert to:
    #
    # [N, 32, 32, 3]
    # --------------------------------------------------------

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

    return images, np.array(labels)


# ============================================================
# LOAD COMPLETE CIFAR-10
# ============================================================

def load_cifar10():

    train_images = []

    train_labels = []

    # --------------------------------------------------------
    # Five training batches
    # --------------------------------------------------------

    for i in range(1, 6):

        file_path = os.path.join(
            DATASET_PATH,
            f"data_batch_{i}"
        )

        images, labels = load_batch(
            file_path
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

    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

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
# CREATE DATA LOADERS
# ============================================================

def create_dataloaders():

    print("\nLoading CIFAR-10...")

    (
        train_images,
        train_labels,
        test_images,
        test_labels
    ) = load_cifar10()

    print(
        f"Total training images: "
        f"{len(train_images):,}"
    )

    print(
        f"Total test images: "
        f"{len(test_images):,}"
    )

    # --------------------------------------------------------
    # Same split used in previous tasks
    #
    # 45,000 training
    # 5,000 validation
    # 10,000 test
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
    # Training augmentation
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
    # Validation/test transform
    # --------------------------------------------------------

    eval_transform = transforms.Compose([

        transforms.ToPILImage(),

        transforms.ToTensor(),

        transforms.Normalize(
            mean,
            std
        )
    ])

    # --------------------------------------------------------
    # Separate datasets
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    print(
        f"Train batches: {len(train_loader)}"
    )

    print(
        f"Validation batches: {len(val_loader)}"
    )

    print(
        f"Test batches: {len(test_loader)}"
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# LOAD TEACHER
# ============================================================

def load_teacher(device):

    print("\nLoading ResNet34 teacher...")

    teacher = ResNet34(
        num_classes=10
    )

    if not os.path.exists(
        TEACHER_CHECKPOINT
    ):

        raise FileNotFoundError(
            "\nTeacher checkpoint not found:\n"
            f"{TEACHER_CHECKPOINT}\n\n"
            "Do NOT start student training until "
            "the Task 1 teacher checkpoint exists."
        )

    checkpoint = torch.load(
        TEACHER_CHECKPOINT,
        map_location=device
    )

    # --------------------------------------------------------
    # Support either:
    #
    # checkpoint["model_state_dict"]
    #
    # or directly a state dict
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ) and "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif isinstance(
        checkpoint,
        dict
    ) and "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    teacher.load_state_dict(
        state_dict
    )

    teacher = teacher.to(device)

    # --------------------------------------------------------
    # Freeze teacher
    # --------------------------------------------------------

    teacher.eval()

    for parameter in teacher.parameters():

        parameter.requires_grad = False

    print(
        "Teacher loaded successfully."
    )

    print(
        "Teacher parameters frozen: True"
    )

    return teacher


# ============================================================
# LOAD STUDENT
# ============================================================

def create_student(device):

    print("\nCreating ternary ResNet18 student...")

    student = TernaryResNet18(
        num_classes=10
    ).to(device)

    parameters = sum(
        p.numel()
        for p in student.parameters()
        if p.requires_grad
    )

    print(
        f"Student trainable parameters: "
        f"{parameters:,}"
    )

    return student


# ============================================================
# KNOWLEDGE DISTILLATION LOSS
# ============================================================

def distillation_loss(
    student_logits,
    teacher_logits,
    targets
):

    # --------------------------------------------------------
    # Standard supervised cross entropy
    # --------------------------------------------------------

    ce_loss = F.cross_entropy(
        student_logits,
        targets,
        label_smoothing=LABEL_SMOOTHING
    )

    # --------------------------------------------------------
    # Teacher/student softened probabilities
    # --------------------------------------------------------

    student_log_probs = F.log_softmax(
        student_logits / TEMPERATURE,
        dim=1
    )

    teacher_probs = F.softmax(
        teacher_logits / TEMPERATURE,
        dim=1
    )

    # --------------------------------------------------------
    # KL divergence
    # --------------------------------------------------------

    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    )

    # Restore temperature scaling
    kd_loss = (
        kd_loss
        *
        (TEMPERATURE ** 2)
    )

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    total_loss = (
        CE_WEIGHT * ce_loss
        +
        KD_WEIGHT * kd_loss
    )

    return (
        total_loss,
        ce_loss,
        kd_loss
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    student,
    teacher,
    train_loader,
    optimizer,
    device,
    epoch
):

    student.train()

    teacher.eval()

    running_loss = 0.0

    running_ce = 0.0

    running_kd = 0.0

    correct = 0

    total = 0

    # --------------------------------------------------------
    # Iterate over training data
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # Teacher forward
        #
        # No gradients.
        # ----------------------------------------------------

        with torch.no_grad():

            teacher_logits = teacher(
                images
            )

        # ----------------------------------------------------
        # Student forward
        #
        # IMPORTANT:
        #
        # ternary_resnet18.py performs QAT here.
        #
        # Every Conv/FC uses:
        #
        #       {-alpha, 0, +alpha}
        # ----------------------------------------------------

        student_logits = student(
            images
        )

        # ----------------------------------------------------
        # KD + CE loss
        # ----------------------------------------------------

        (
            loss,
            ce_loss,
            kd_loss
        ) = distillation_loss(
            student_logits,
            teacher_logits,
            targets
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Gradient clipping
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            GRAD_CLIP
        )

        # ----------------------------------------------------
        # Update student
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = targets.size(0)

        running_loss += (
            loss.item()
            *
            batch_size
        )

        running_ce += (
            ce_loss.item()
            *
            batch_size
        )

        running_kd += (
            kd_loss.item()
            *
            batch_size
        )

        predictions = (
            student_logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += batch_size

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            batch_idx == 0
            or
            (batch_idx + 1) % 100 == 0
            or
            batch_idx == len(train_loader) - 1
        ):

            current_acc = (
                100.0
                *
                correct
                /
                total
            )

            print(
                f"\rEpoch {epoch:03d} | "
                f"Batch {batch_idx + 1:03d}/"
                f"{len(train_loader):03d} | "
                f"Loss {loss.item():.4f} | "
                f"Acc {current_acc:.2f}%",
                end=""
            )

    print()

    epoch_loss = (
        running_loss
        /
        total
    )

    epoch_ce = (
        running_ce
        /
        total
    )

    epoch_kd = (
        running_kd
        /
        total
    )

    epoch_acc = (
        100.0
        *
        correct
        /
        total
    )

    return (
        epoch_loss,
        epoch_acc,
        epoch_ce,
        epoch_kd
    )


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

    running_loss = 0.0

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

        logits = model(
            images
        )

        loss = F.cross_entropy(
            logits,
            targets
        )

        batch_size = targets.size(0)

        running_loss += (
            loss.item()
            *
            batch_size
        )

        predictions = (
            logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == targets
        ).sum().item()

        total += batch_size

    loss = (
        running_loss
        /
        total
    )

    accuracy = (
        100.0
        *
        correct
        /
        total
    )

    return (
        loss,
        accuracy
    )


# ============================================================
# CHECK ALPHA VALUES
# ============================================================

@torch.no_grad()
def print_alpha_statistics(
    model
):

    print("\nAlpha statistics:")

    count = 0

    for name, module in model.named_modules():

        if hasattr(
            module,
            "alpha"
        ):

            alpha_value = (
                module.alpha.item()
            )

            print(
                f"{name:35s} "
                f"alpha={alpha_value:.6f}"
            )

            count += 1

    print(
        f"Alpha parameters checked: "
        f"{count}"
    )


# ============================================================
# VERIFY QAT
# ============================================================

@torch.no_grad()
def verify_qat(
    model
):

    print(
        "\nVerifying QAT ternary weights..."
    )

    all_valid = True

    checked = 0

    for name, module in model.named_modules():

        if not isinstance(
            module,
            (
                # Import classes through module
                # type checking below
                type(model.conv1),
            )
        ):

            # We still need to identify TernaryLinear
            # and TernaryConv2d safely.
            pass

    # --------------------------------------------------------
    # Directly use the model's own verification function
    # --------------------------------------------------------

    try:

        from ternary_resnet18 import (
            check_ternary_weights
        )

        all_valid = check_ternary_weights(
            model,
            verbose=False
        )

    except Exception as error:

        print(
            "QAT verification error:",
            error
        )

        all_valid = False

    print(
        "Ternary QAT:",
        "PASS"
        if all_valid
        else "FAIL"
    )

    return all_valid


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    student,
    optimizer,
    scheduler,
    epoch,
    val_accuracy,
    train_accuracy,
    path
):

    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {

        "epoch": epoch,

        "model_state_dict":
            student.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict(),

        "val_accuracy":
            val_accuracy,

        "train_accuracy":
            train_accuracy
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    print("=" * 70)

    print(
        "TASK 3: TERNARY RESNET18 + "
        "KNOWLEDGE DISTILLATION + QAT"
    )

    print("=" * 70)

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
    # Teacher
    # --------------------------------------------------------

    teacher = load_teacher(
        device
    )

    # --------------------------------------------------------
    # Student
    # --------------------------------------------------------

    student = create_student(
        device
    )

    # --------------------------------------------------------
    # Optimizer parameter groups
    #
    # Alpha receives a smaller learning rate.
    # --------------------------------------------------------

    normal_parameters = []

    alpha_parameters = []

    for name, parameter in student.named_parameters():

        if not parameter.requires_grad:

            continue

        if (
            "log_alpha" in name
        ):

            alpha_parameters.append(
                parameter
            )

        else:

            normal_parameters.append(
                parameter
            )

    print(
        "\nOptimizer parameters:"
    )

    print(
        f"Normal parameters: "
        f"{len(normal_parameters)}"
    )

    print(
        f"Alpha parameters: "
        f"{len(alpha_parameters)}"
    )

    optimizer = torch.optim.SGD(

        [
            {
                "params":
                    normal_parameters,

                "lr":
                    LEARNING_RATE
            },

            {
                "params":
                    alpha_parameters,

                "lr":
                    LEARNING_RATE * 0.5
            }
        ],

        momentum=MOMENTUM,

        weight_decay=WEIGHT_DECAY,

        nesterov=True
    )

    # --------------------------------------------------------
    # Cosine learning-rate schedule
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=EPOCHS,

        eta_min=1e-4
    )

    # --------------------------------------------------------
    # Training variables
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    best_epoch = 0

    best_train_accuracy = 0.0

    # --------------------------------------------------------
    # Initial QAT check
    # --------------------------------------------------------

    verify_qat(
        student
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    print("\n" + "=" * 70)

    print(
        f"Starting {EPOCHS}-epoch training..."
    )

    print("=" * 70)

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Learning rate: "
            f"{current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        (
            train_loss,
            train_accuracy,
            ce_loss,
            kd_loss
        ) = train_one_epoch(

            student,

            teacher,

            train_loader,

            optimizer,

            device,

            epoch
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            student,
            val_loader,
            device
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Print epoch results
        # ----------------------------------------------------

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.2f}%"
        )

        print(
            f"CE Loss: {ce_loss:.4f}"
        )

        print(
            f"KD Loss: {kd_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy:.2f}%"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            best_epoch = epoch

            best_train_accuracy = (
                train_accuracy
            )

            save_checkpoint(

                student,

                optimizer,

                scheduler,

                epoch,

                val_accuracy,

                train_accuracy,

                STUDENT_CHECKPOINT
            )

            print(
                "\n*** New best student checkpoint saved! ***"
            )

            print(
                STUDENT_CHECKPOINT
            )

        # ----------------------------------------------------
        # Periodic QAT check
        # ----------------------------------------------------

        if (
            epoch == 1
            or
            epoch % 10 == 0
        ):

            verify_qat(
                student
            )

            print_alpha_statistics(
                student
            )

        # ----------------------------------------------------
        # GPU memory
        # ----------------------------------------------------

        if torch.cuda.is_available():

            memory_used = (
                torch.cuda.memory_allocated()
                /
                (1024 ** 3)
            )

            memory_reserved = (
                torch.cuda.memory_reserved()
                /
                (1024 ** 3)
            )

            print(
                f"\nGPU memory: "
                f"{memory_used:.2f} GB allocated / "
                f"{memory_reserved:.2f} GB reserved"
            )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "Loading best student checkpoint..."
    )

    print("=" * 70)

    best_checkpoint = torch.load(
        STUDENT_CHECKPOINT,
        map_location=device
    )

    if (
        isinstance(
            best_checkpoint,
            dict
        )
        and
        "model_state_dict" in best_checkpoint
    ):

        student.load_state_dict(
            best_checkpoint[
                "model_state_dict"
            ]
        )

    else:

        student.load_state_dict(
            best_checkpoint
        )

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\nEvaluating best student on test set..."
    )

    test_loss, test_accuracy = evaluate(
        student,
        test_loader,
        device
    )

    train_loss, final_train_accuracy = evaluate(
        student,
        train_loader,
        device
    )

    # --------------------------------------------------------
    # Train/test gap
    # --------------------------------------------------------

    train_test_gap = (
        final_train_accuracy
        -
        test_accuracy
    )

    # --------------------------------------------------------
    # Final QAT verification
    # --------------------------------------------------------

    qat_pass = verify_qat(
        student
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TASK 3 FINAL RESULTS"
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
        f"Ternary QAT: "
        f"{'PASS' if qat_pass else 'FAIL'}"
    )

    print(
        "\nStudent checkpoint:"
    )

    print(
        STUDENT_CHECKPOINT
    )

    print("=" * 70)

    print(
        "\nTask 3 training completed."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()