import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from resnet34_cifar import ResNet34


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 100
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "resnet34_cifar10_teacher_best.pth"
)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# EVALUATION FUNCTION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("CIFAR-10 RESNET34 TEACHER TRAINING")
    print("=" * 65)

    print("\nDevice:", DEVICE)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

        print(
            "GPU Memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / (1024 ** 3),
                2
            ),
            "GB"
        )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_loader, val_loader, test_loader = (
        create_dataloaders()
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating ResNet34 teacher...")

    model = ResNet34(
        num_classes=10
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        label_smoothing=LABEL_SMOOTHING
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # --------------------------------------------------------
    # Cosine learning-rate scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Training history
    # --------------------------------------------------------

    history = {

        "train_loss": [],
        "train_accuracy": [],

        "val_loss": [],
        "val_accuracy": [],

        "learning_rate": []
    }

    best_val_accuracy = 0.0
    best_epoch = 0

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(1, EPOCHS + 1):

        start_time = time.time()

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            DEVICE
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        epoch_time = time.time() - start_time

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history["train_loss"].append(
            train_loss
        )

        history["train_accuracy"].append(
            train_accuracy
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_accuracy"].append(
            val_accuracy
        )

        history["learning_rate"].append(
            current_lr
        )

        # ----------------------------------------------------
        # Validation gap
        # ----------------------------------------------------

        generalization_gap = (
            train_accuracy - val_accuracy
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "scheduler_state_dict":
                        scheduler.state_dict(),

                    "best_val_accuracy":
                        best_val_accuracy,

                    "train_accuracy":
                        train_accuracy,

                    "val_accuracy":
                        val_accuracy,

                    "history":
                        history
                },
                BEST_MODEL_PATH
            )

            best_marker = "  <-- BEST"

        else:

            best_marker = ""

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        print(
            f"Epoch [{epoch:03d}/{EPOCHS}] "
            f"LR: {current_lr:.5f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"Gap: {generalization_gap:.2f}% | "
            f"Time: {epoch_time:.1f}s"
            f"{best_marker}"
        )

    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print("\n" + "=" * 65)
    print("LOADING BEST TEACHER CHECKPOINT")
    print("=" * 65)

    checkpoint = torch.load(
        BEST_MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    train_loss, train_accuracy = evaluate(
        model,
        train_loader,
        criterion,
        DEVICE
    )

    final_gap = (
        train_accuracy - test_accuracy
    )

    print("\n========== FINAL TEACHER RESULTS ==========")

    print(
        f"Best Epoch       : {best_epoch}"
    )

    print(
        f"Best Val Accuracy: {best_val_accuracy:.2f}%"
    )

    print(
        f"Train Accuracy   : {train_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy    : {test_accuracy:.2f}%"
    )

    print(
        f"Train/Test Gap   : {final_gap:.2f}%"
    )

    print(
        f"Test Loss        : {test_loss:.4f}"
    )

    print(
        f"\nTeacher saved to:"
    )

    print(
        BEST_MODEL_PATH
    )

    print("============================================")


if __name__ == "__main__":

    main()