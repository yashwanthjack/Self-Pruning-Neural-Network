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

# --- CONFIGURATION ---
EPOCHS = 20
BATCH_SIZE = 512
LAMBDA_VAL = 1e-6 

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
        # 1. Standard parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 2. RULE: gate_scores must have the EXACT SAME SHAPE as the weight tensor
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, 0)
        # Initialize gates slightly positive
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, x):
        # 3a. Transformation to gate_scores using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # 3b. Calculate pruned weights by element-wise multiplication
        pruned_weights = self.weight * gates
        
        # 3c. Perform standard linear layer operation
        return F.linear(x, pruned_weights, self.bias)

# --- THE "ULTIMATE" MLP ARCHITECTURE (Assignment Compliant) ---
class OptimizedPrunableMLP(nn.Module):
    def __init__(self):
        super(OptimizedPrunableMLP, self).__init__()
        self.flatten = nn.Flatten()
        
        # Consistent with "Standard Feed-Forward" requirement
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
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    total_w, pruned_w = 0, 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                total_w += gates.numel()
                pruned_w += torch.sum(gates < threshold).item()
                
    accuracy = 100 * correct / total
    sparsity = 100 * pruned_w / total_w if total_w > 0 else 0
    return accuracy, sparsity

def main():
    parser = argparse.ArgumentParser(description="Optimized Assignment MLP")
    parser.add_argument('--epochs', type=int, default=20, help='Total epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lambda-val', type=float, default=1e-6, help='Sparsity penalty')
    args = parser.parse_known_args()[0]

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Optimized Assignment Model on: {device}")
    
    save_dir = "results_optimized"
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = OptimizedPrunableMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Differential LR for faster pruning
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'gate_scores' not in n], 'lr': 0.001},
        {'params': [p for n, p in model.named_parameters() if 'gate_scores' in n], 'lr': 0.01}
    ])

    print(f"\nTraining with Strictly Compliant Lambda = {args.lambda_val}...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # RULE: Total Loss = ClassificationLoss + Lambda * SparsityLoss
            total_loss = criterion(outputs, labels) + args.lambda_val * get_sparsity_loss(model)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        acc, spa = evaluate_model(model, testloader, device)
        print(f"Epoch {epoch+1:02d} | Loss: {running_loss/len(trainloader):.4f} | Acc: {acc:.2f}% | Sparsity: {spa:.2f}% | Time: {time.time()-start_time:.1f}s")

    # Save distribution plot
    all_gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            all_gates.extend(torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten())
    plt.figure(figsize=(10,6))
    plt.hist(all_gates, bins=50, color='blue', alpha=0.7)
    plt.title(f"Weight-level Gate Distribution (Lambda={args.lambda_val})")
    plt.savefig(os.path.join(save_dir, 'gate_distribution.png'))
    print(f"\nResults saved to {save_dir}/")

if __name__ == "__main__":
    main()
