import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import os
import time
import argparse
import random
import numpy as np
import csv
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class PrunableLinear(nn.Module):
    """
    Part 1 implementation: A custom prunable linear layer.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(PrunableLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 1. Standard weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # 2. Crucially, a second parameter tensor with the exact same shape as the weight tensor.
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        # Initialize slightly positive so starting gradients exist
        nn.init.constant_(self.gate_scores, 0.5)

    def forward(self, input):
        # 3a. Transformation to gate_scores using Sigmoid
        gates = torch.sigmoid(self.gate_scores)
        
        # 3b. Calculate pruned weights by element-wise multiplication
        pruned_weights = self.weight * gates
        
        # 3c. Perform standard linear layer operation
        return F.linear(input, pruned_weights, self.bias)

class PrunableNet(nn.Module):
    """
    Standard feed-forward neural network for CIFAR-10.
    """
    def __init__(self):
        super(PrunableNet, self).__init__()
        self.flatten = nn.Flatten()
        # CIFAR-10 images are 3x32x32 = 3072 input features
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_sparsity_loss(model):
    """
    Part 2 formulation: L1 norm of all gate values.
    Since gates are positive, this is simply the sum.
    """
    loss = 0.0
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            loss = loss + torch.sum(gates)
    return loss

def evaluate_model(model, dataloader, device, threshold=1e-2):
    """
    Part 3 evaluation logic. Returns accuracy and exact sparsity percentage.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    
    # Calculate exact sparsity using the assignment requirements
    total_weights = 0
    pruned_weights = 0
    
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores)
                # Count gates below the small threshold
                hard_gates = (gates < threshold)
                total_weights += gates.numel()
                pruned_weights += torch.sum(hard_gates).item()
                
    sparsity_level = 100 * pruned_weights / total_weights if total_weights > 0 else 0
    
    return accuracy, sparsity_level

def plot_gates_distribution(model, filename):
    all_gates = []
    with torch.no_grad():
         for module in model.modules():
            if isinstance(module, PrunableLinear):
                gates = torch.sigmoid(module.gate_scores).flatten().cpu().numpy()
                all_gates.extend(gates)
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of Final Gate Values')
    plt.xlabel('Gate Value (0 = Pruned, 1 = Kept)')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Self-Pruning MLP loosely based on CIFAR-10")
    parser.add_argument('--epochs', type=int, default=15, help='Total epochs to train')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for dataloaders')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # We must formulate tiny lambda penalties because the loss is an un-normalized SUM of ~1.6M values.
    lambdas = [0.0, 1e-5, 5e-5]
    
    summary_results = []

    for l_val in lambdas:
        print(f"\n--- Training PrunableNet with Lambda = {l_val} ---")
        model = PrunableNet().to(device)
        criterion = nn.CrossEntropyLoss()
        
        # High lr on gates vs weights to facilitate pruning effectively
        optimizer = optim.Adam([
            {'params': [p for n, p in model.named_parameters() if 'gate_scores' not in n], 'lr': 0.001},
            {'params': [p for n, p in model.named_parameters() if 'gate_scores' in n], 'lr': 0.01}
        ])

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            start_time = time.time()
            
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                
                classification_loss = criterion(outputs, labels)
                sparsity_loss = get_sparsity_loss(model)
                total_loss = classification_loss + l_val * sparsity_loss
                
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
            
            epoch_time = time.time() - start_time
            train_loss = running_loss / len(trainloader)
            
            # Validation at end of epoch
            test_acc, sparsity = evaluate_model(model, testloader, device)
            print(f"Epoch {epoch+1:02d}/{args.epochs} - Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Test Acc: {test_acc:.2f}% | Sparsity: {sparsity:.2f}%")

        # Save results mapping
        test_acc, final_sparsity = evaluate_model(model, testloader, device)
        summary_results.append((l_val, test_acc, final_sparsity))
        
        if l_val > 0:
            plot_gates_distribution(model, os.path.join(save_dir, f'gates_dist_lambda_{l_val}.png'))

    print("\n--- Summary ---")
    print("| Lambda | Test Accuracy (%) | Sparsity Level (%) |")
    print("|--------|-------------------|--------------------|")
    for l_val, acc, sparsity in summary_results:
        print(f"| {l_val:<6} | {acc:<17.2f} | {sparsity:<18.2f} |")

    summary_csv = os.path.join(save_dir, 'summary_results.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Lambda', 'Test Accuracy (%)', 'Sparsity Level (%)'])
        for row in summary_results:
            writer.writerow(row)
            
if __name__ == "__main__":
    main()


