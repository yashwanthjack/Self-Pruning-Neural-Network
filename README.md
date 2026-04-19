# The Self-Pruning Neural Network

This repository contains a neural network implementation for the CIFAR-10 dataset that actively learns to structurally prune itself during training by minimizing the L1 norm of learnable gate parameters.

## Problem Context
Deploying large neural networks is often constrained by memory and computational budgets. Instead of pruning weights post-training as an arbitrary step, this model associates each weight with a learnable gate scalar (between 0 and 1). The total training loss is augmented with an L1 regularization penalty on the gates, actively encouraging the architecture to shut down unnecessary connections on the fly.

## 1. Why an L1 Penalty Encourages Sparsity
If we only train the network using standard Cross-Entropy loss, the optimizer focuses strictly on minimizing classification error, leaving the active `gate_scores` near 1.0 since there is no incentive to shut weights off.

By adding an L1 penalty (the un-normalized sum of all positive `sigmoid(gate_scores)`) to the loss function, we introduce a competitive objective. The optimizer must now trade off between reducing classification error and reducing the raw magnitude of the active gates. Because the derivative of the L1 norm provides a constant geometric pressure toward zero (unlike L2), it aggressively forces the least-important gates to shrink until they reach 0. The hyperparameter `λ` explicitly controls the severity of this shrinkage pressure.

## 2. Experimental Results

The table below summarizes the trade-off between sparsity penalty (λ) and the resulting test accuracy on the CIFAR-10 dataset using our optimized feed-forward architecture (3072 → 1024 → 512 → 256 → 10 with BatchNorm and Dropout).

| Lambda | Test Accuracy (%) | Sparsity Level (%) | Observation |
|--------|-------------------|--------------------|-------------|
| 0.0    | 57.87             | 0.00               | Baseline — No pruning pressure |
| 1e-06  | 59.08             | 5.22               | Minimal pruning, accuracy preserved |
| **1e-05** | **59.11**      | **49.95**          | **Sweet Spot — 50% pruned, +1.2% Accuracy boost** |

At λ = 10⁻⁵, the network autonomously pruned **49.95%** of its weights while actually improving test accuracy by **1.24%**. This is a remarkable result that demonstrates how the gated L1 penalty acts as a superior regularizer, eliminating noisy "lottery tickets" while preserving the essential sub-network. This is consistent with the **Lottery Ticket Hypothesis** (Frankle & Carlin, 2019).

## 3. Gate Distribution Analysis
The generated histogram plots show a dominant spike at **gate value ≈ 0**, confirming the L1 penalty successfully drives unimportant gates to zero. The remaining gates form a long tail, representing the optimal sparse sub-network identified by the model.

## 4. Architecture Details

| Component | Specification |
|-----------|--------------|
| Input     | CIFAR-10 (3×32×32 = 3072 features) |
| Hidden Layers | 1024 → 512 → 256 (with BatchNorm + Dropout) |
| Output    | 10 classes |
| Gate Shape | Identical to weight tensor (element-wise pruning) |
| Loss      | CrossEntropy + λ × Σ sigmoid(gate_scores) |
| Optimizer | Adam (differential LR: 0.001 weights, 0.01 gates) |

## Usage

```bash
# Default training (runs 3 lambdas sequentially)
python prunable_network.py
```

## Repository Structure
```
├── prunable_network.py          # Main assignment-compliant implementation
├── results/                     # Generated plots and CSVs
└── README.md                    # This report
```
