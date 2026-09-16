# CIFAR-10 Model Comparison

This project compares three CIFAR-10 image-classification models:

1. **ResNet-34 teacher**: full-precision FP32 model trained from scratch.
2. **ResNet-18 baseline**: full-precision FP32 model trained from scratch without knowledge distillation.
3. **Ternary ResNet-18 student**: a ternary-weight model trained with knowledge distillation and quantization-aware training (QAT).

The main analysis is in [`Model_Comparison_Task4.ipynb`](Model_Comparison_Task4.ipynb). It creates comparison tables, plots, a CSV summary, and ternary-weight compression statistics under `results/`.

## Results Summary

| Model | Training setup | Best epoch | Train accuracy | Evaluation accuracy | Accuracy gap | Parameters |
|---|---|---:|---:|---:|---:|---:|
| ResNet-34 Teacher | FP32 | 96 | 99.98% | 94.91% | 5.07 pp | 21,282,122 |
| ResNet-18 Baseline | FP32 | 98 | 100.00% | 94.58% | 5.42 pp | 11,173,983 |
| Ternary ResNet-18 Student | Ternary + KD + QAT | 139 | 91.03% | 88.14% | 2.89 pp | 11,173,983 |

The ternary student has lower accuracy than both FP32 models, but it uses ternary weights and has the smallest train-to-evaluation accuracy gap among the three models.

## Project Structure

```text
C:\Hamza_atdl\
├── Model_Comparison_Task4.ipynb   # Tables, plots, and final analysis
├── dataset.py                     # CIFAR-10 batch loading and transforms
├── resnet34_cifar.py              # ResNet-34 architecture
├── ternary_resnet18.py            # Ternary ResNet-18 architecture
├── train_teacher.py               # Train the ResNet-34 teacher
├── train_resnet18_baseline.py     # Train the FP32 ResNet-18 baseline
├── train_student_kd.py            # Train the ternary KD/QAT student
├── load_cifar.py                  # CIFAR-10 loading utility
├── checkpoints/                   # Saved best model checkpoints
├── cifar-10-batches-py/           # Local CIFAR-10 dataset
└── results/                       # Generated plots and summaries
```

## Requirements

- Python 3.9 or newer
- PyTorch
- torchvision
- NumPy
- pandas
- Matplotlib
- Jupyter or VS Code with the Jupyter extension

Install the Python packages in the active environment with:

```powershell
python -m pip install torch torchvision numpy pandas matplotlib jupyter
```

The training scripts use CUDA automatically when it is available and otherwise run on the CPU.

## Dataset

The scripts expect the CIFAR-10 Python batch files at:

```text
C:\Hamza_atdl\cifar-10-batches-py
```

The loader applies CIFAR-10 normalization using mean `(0.4914, 0.4822, 0.4465)` and standard deviation `(0.2470, 0.2435, 0.2616)`. Update `DATA_DIR` or `DATASET_PATH` in the scripts if the dataset is stored elsewhere.

## Training

Run these commands from `C:\Hamza_atdl`:

```powershell
python train_teacher.py
python train_resnet18_baseline.py
python train_student_kd.py
```

The student training script loads the teacher checkpoint and saves the best student checkpoint. The expected checkpoint files are:

```text
checkpoints/resnet34_cifar10_teacher_best.pth
checkpoints/resnet18_cifar10_baseline_best.pth
checkpoints/ternary_resnet18_kd_best.pth
```

The student combines cross-entropy loss with teacher-guided distillation. Its ternary convolution and fully connected weights are represented during forward passes using the values `{-alpha, 0, +alpha}`.

## Notebook Analysis

Open and run [`Model_Comparison_Task4.ipynb`](Model_Comparison_Task4.ipynb) after training, or use it to reproduce the recorded comparison and plots. The notebook generates:

- Evaluation-accuracy comparison
- Training versus evaluation accuracy
- Generalization-gap comparison
- Parameter-count comparison
- Best-epoch comparison
- Ternary-student accuracy
- Ternary weight distribution
- Student sparsity
- Theoretical FP32 versus ternary storage
- Theoretical compression ratio

The ternary checkpoint analysis section reads `checkpoints/ternary_resnet18_kd_best.pth` and writes `ternary_analysis.txt` plus the ternary-analysis plots to `results/`.

## Generated Results

Important output files include:

```text
results/FINAL_Task4_Three_Model_Comparison.csv
results/ternary_analysis.txt
results/01_accuracy_comparison.png
results/02_train_vs_evaluation_accuracy.png
results/03_generalization_gap.png
results/04_parameter_count.png
results/05_best_epoch.png
results/07_student_accuracy.png
results/09_ternary_weight_distribution.png
results/10_student_sparsity.png
results/11_theoretical_storage.png
results/12_theoretical_compression_ratio.png
```

The theoretical compression calculation assumes 32 bits per FP32 weight and 2 bits per ternary weight. It does not include metadata, scaling factors, indexes, or runtime overhead.

## Reproducibility Notes

- The FP32 baseline and student scripts use a fixed seed of `42`.
- Results recorded in the notebook are summary values and are not recomputed automatically from training logs.
- Paths are currently Windows-specific and should be changed when moving the project to another machine.