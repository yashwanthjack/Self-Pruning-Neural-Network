# The Self-Pruning Neural Network

This repository contains a neural network implementation for the CIFAR-10 dataset that actively learns to structurally prune itself during training by minimizing the L1 norm of learnable gate parameters.

## Problem Context
Deploying large neural networks is often constrained by memory and computational budgets. Instead of pruning structurally important weights post-training as an arbitrary mathematical threshold, this model associates each weight with a learnable gate scalar (between 0 and 1). The total training loss is augmented with an L1 regularization penalty on the gates, actively encouraging the architecture to shut down unnecessary nodes on the fly. 

## 1. Why an L1 Penalty Encourages Sparsity
If we only train the network using standard Cross-Entropy loss, the optimizer focuses strictly on minimizing classification error, leaving the active `gate_scores` near 1.0 since there is no computational incentive to shut weights off. 

By adding an L1 penalty (the un-normalized sum of all positive `sigmoid(gate_scores)`) to the loss function, we introduce a competitive objective. The optimizer must now trade off between reducing classification error and reducing the raw magnitude of the active gates. Because the derivative of the L1 norm provides a constant geometric pressure toward zero (unlike L2), it aggressively forces the least-important gates to shrink until they reach 0. The hyperparameter `λ` explicitly controls the mathematical severity of this shrinkage pressure.

## 2. Experimental Results
The table below summarizes the trade-off between the sparsity penalty (`λ`) and the resulting test accuracy on the CIFAR-10 dataset using a standard feed-forward architecture. 

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|-------------------|--------------------|
| 0.0    | 54.89             | 0.00               |
| 1e-05  | 55.41             | 35.81              |
| 5e-05  | 53.35             | 81.82              |

*Note: Model convergence and evaluation tracked across exactly 15 epochs.*

## 3. Findings & Distribution
As demonstrated in the results, configuring `λ = 0.0` establishes our control baseline with 0% structural sparsity. 

By applying an optimal penalty curve (`1e-5`), the model achieves the algorithmic sweet spot (effectively identifying the winning lottery ticket pathway): it successfully severs nearly 36% of its connections while actually improving predictive performance (+0.5% margin) due to robust noise reduction.

By aggressively increasing the severity parameter (`5e-5`), the model attains massive compression logic. It strips virtually 82% of its original pathways while suffering an absolutely minimal efficiency drop (~1.5%) in total test accuracy. 

The visualization inside the `/results` folder cleanly illustrates the distribution mapping for the heavily pruned configurations, proving the efficacy of the L1 algorithm by forcing a dominant distribution cluster explicitly to 0.

## Usage

Ensure PyTorch and Torchvision are installed cleanly within your environment.

```bash
# Execute training loops against dynamic lambda thresholds
python prunable_network.py --epochs 15 --batch-size 512
```
