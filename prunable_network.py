import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import os
import time
import random
import numpy as np
import csv
import matplotlib.pyplot as plt
import argparse

# --- ASSIGNMENT CONFIGURATION ---
EPOCHS = 20
BATCH_SIZE = 512
# Note: Because the loss is a SUM of ~4 million gates, Lambda must be very small.
# 1e-6 is "Standard", 5e-6 is "Aggressive".
LAMBDA_VAL = 1e-05

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- PART 1: THE PRUNABLE LINEAR LAYER (Strictly Compliant) ---
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        # Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # RULE: gate_scores must have the EXACT SAME SHAPE as the weight tensor
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.5) 

    def forward(self, x):
        # RULE: Apply Sigmoid to gate_scores
        gates = torch.sigmoid(self.gate_scores)
        
        # RULE: Calculate pruned weights (element-wise multiplication)
        pruned_weights = self.weight * gates
        
        # RULE: Perform standard linear layer operation with pruned weights
        return F.linear(x, pruned_weights, self.bias)

# --- THE "ULTIMATE" MLP ARCHITECTURE ---
class UltimatePrunableMLP(nn.Module):
    def __init__(self):
        super(UltimatePrunableMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Architecture: Wide & Deep MLP (3072 -> 1024 -> 512 -> 256 -> 10)
        self.fc1 = PrunableLinear(3072, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        
        self.fc2 = PrunableLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        
        self.fc3 = PrunableLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.fc4 = PrunableLinear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

def get_sparsity_loss(model):
    # RULE: SparsityLoss is simply the SUM of all gate values
    loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            loss += torch.sum(torch.sigmoid(module.gate_scores))
    return loss

def evaluate_model(model, dataloader, device, threshold=1e-2):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, prd = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (prd == labels).sum().item()
    
    # Calculate Sparsity Level
    total_w, pruned_w = 0, 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_w += gates.numel()
                pruned_w += torch.sum(gates < threshold).item()
                
    return 100 * correct / total, 100 * pruned_w / total_w

def plot_gates_distribution(model, filename, lambda_val):
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            all_gates.extend(torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten())
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Weight-level Gate Distribution (Lambda={lambda_val})")
    plt.xlabel("Gate Value (0 = Pruned, 1 = Kept)")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning MLP on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=20, help='Total epochs to train')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    args = parser.parse_known_args()[0]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running assignment-compliant training on: {device}")

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    # Data prep
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # REQUIREMENT: Compare at least three different values of Lambda
    lambdas = [0.0, 1e-06, 1e-05]
    summary_results = []

    for l_val in lambdas:
        print(f"\n{'='*60}")
        print(f"--- Training with Lambda = {l_val} ---")
        print(f"{'='*60}")

        set_seed(42)
        model = UltimatePrunableMLP().to(device)
        criterion = nn.CrossEntropyLoss()

        # Differential LR: Gates learn faster
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'gate_scores' not in n], 'lr': 0.001},
            {'params': [p for n, p in model.named_parameters() if 'gate_scores' in n], 'lr': 0.01}
        ])

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            start = time.time()
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                # RULE: Total Loss = ClassificationLoss + Lambda * SparsityLoss
                total_loss = criterion(outputs, labels) + l_val * get_sparsity_loss(model)

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()

            acc, spa = evaluate_model(model, testloader, device)
            print(f"Epoch {epoch+1:02d}/{args.epochs} | Loss: {running_loss/len(trainloader):.4f} | Acc: {acc:.2f}% | Sparsity: {spa:.2f}% | Time: {time.time()-start:.1f}s")

        # Final evaluation for this lambda
        final_acc, final_spa = evaluate_model(model, testloader, device)
        summary_results.append((l_val, final_acc, final_spa))

        # Save gate distribution plot for each lambda
        if l_val > 0:
            plot_gates_distribution(model, os.path.join(save_dir, f'gates_dist_lambda_{l_val}.png'), l_val)

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}")
    print(f"| {'Lambda':<10} | {'Test Accuracy (%)':<18} | {'Sparsity Level (%)':<19} |")
    print(f"|{'-'*12}|{'-'*20}|{'-'*21}|")
    for l_val, acc, spa in summary_results:
        print(f"| {l_val:<10} | {acc:<18.2f} | {spa:<19.2f} |")

    # Save summary to CSV
    summary_csv = os.path.join(save_dir, 'summary_results.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda', 'Test Accuracy (%)', 'Sparsity Level (%)'])
        for row in summary_results:
            writer.writerow(row)

    print(f"\nAll results saved to {save_dir}/")

if __name__ == "__main__":
    main()

