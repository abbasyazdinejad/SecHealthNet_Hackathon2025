# health_federated_learning.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
from torchvision import transforms
from copy import deepcopy
import argparse

# 1. Set up the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. Differential Privacy functions
def apply_dp(model, noise_scale=1e-3):
    """Add Gaussian noise to model gradients for differential privacy"""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(0, noise_scale, size=param.grad.shape).to(param.device)
            param.grad += noise

# 3. Local training with DP
def local_train_dp(model, dataloader, epochs=1, lr=0.001, dp=True, noise_scale=1e-3):
    """Train a local model with differential privacy"""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for x, y in dataloader:
            x, y = x.to(device), y.squeeze().to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            if dp:
                apply_dp(model, noise_scale=noise_scale)
            optimizer.step()

# 4. Model Weight Utilities
def get_model_weights(model):
    """Get a copy of model weights as a state dict"""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def set_model_weights(model, weights):
    """Set model weights from a state dict"""
    model.load_state_dict(weights)

def average_weights(weights_list):
    """Average multiple model weights together (federated aggregation)"""
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum(w[key] for w in weights_list) / len(weights_list)
    return avg_weights

# 5. Model Evaluation
def evaluate_model(model, dataloader):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.squeeze().to(device)
            preds = model(x)
            _, predicted = torch.max(preds, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    return correct / total

# 6. Visualization Functions
def plot_accuracy(round_accuracies, save_path=None):
    """Plot training accuracy progress by federated round"""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(round_accuracies) + 1), round_accuracies, marker='o', color='green', linewidth=2)
    plt.title("Federated Learning with Privacy | Healthcare Data", fontsize=14)
    plt.xlabel("Federated Round", fontsize=12)
    plt.ylabel("Global Accuracy", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.xticks(range(1, len(round_accuracies) + 1))
    plt.yticks(np.linspace(0, 1, 11))

    for i, acc in enumerate(round_accuracies):
        plt.text(i + 1, acc + 0.03, f"Round {i+1}\nAcc: {acc:.2f}", ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_sample_images(dataset, num_samples=8):
    """Display sample images from the dataset"""
    sample_loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    data_iter = iter(sample_loader)
    images, labels = next(data_iter)

    plt.figure(figsize=(10, 2))
    for idx in range(num_samples):
        plt.subplot(1, num_samples, idx+1)
        plt.imshow(np.transpose(images[idx], (1, 2, 0)))
        plt.axis('off')
    plt.suptitle("Sample PathMNIST Images")
    plt.show()

# 7. Main Training Loop
def federated_learning(num_clients=5, num_rounds=5, local_epochs=1, dp_noise=1e-3, enable_dp=True, 
                        batch_size=64, lr=0.001, save_results=False):
    """Run federated learning across multiple clients with privacy protection"""
    # Load dataset
    data_flag = 'pathmnist'
    download = True
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    train_dataset = DataClass(split='train', transform=transform, download=download)
    test_dataset = DataClass(split='test', transform=transform, download=download)
    
    # Display some sample images
    plot_sample_images(train_dataset)

    # Split dataset into hospitals (clients)
    total_samples = len(train_dataset)
    base_size = total_samples // num_clients
    split_sizes = [base_size] * (num_clients - 1)
    split_sizes.append(total_samples - sum(split_sizes))

    hospital_datasets = random_split(train_dataset, split_sizes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size*2)

    # Initialize global model
    global_model = CNN().to(device)
    round_accuracies = []

    # Federated learning rounds
    for round_num in range(num_rounds):
        print(f"\nFederated Round {round_num+1} | Differential Privacy: {'Enabled' if enable_dp else 'Disabled'}")
        
        local_weights = []

        # Train local models at each hospital
        for client_idx in range(num_clients):
            # Initialize local model with global weights
            local_model = deepcopy(global_model)
            local_loader = DataLoader(hospital_datasets[client_idx], batch_size=batch_size, shuffle=True)
            
            # Train local model
            local_train_dp(
                local_model, 
                local_loader, 
                epochs=local_epochs, 
                lr=lr,
                dp=enable_dp, 
                noise_scale=dp_noise
            )
            
            # Collect local model weights
            local_weights.append(get_model_weights(local_model))

        # Aggregate model weights (FedAvg)
        global_weights = average_weights(local_weights)
        set_model_weights(global_model, global_weights)

        # Evaluate global model
        accuracy = evaluate_model(global_model, test_loader)
        round_accuracies.append(accuracy)
        
        print(f"Global Accuracy after Round {round_num+1}: {accuracy:.4f} | "
              f"{'🔒 Secure' if enable_dp else '⚠️ Not Private'}")

    # Visualize results
    plot_accuracy(round_accuracies, 'federated_accuracy.png' if save_results else None)
    
    return global_model, round_accuracies

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Federated Learning for Healthcare with Privacy')
    parser.add_argument('--num_clients', type=int, default=5, help='Number of hospitals/clients')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--local_epochs', type=int, default=1, help='Number of local epochs per round')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dp_noise', type=float, default=1e-3, help='Differential privacy noise scale')
    parser.add_argument('--no_dp', action='store_true', help='Disable differential privacy')
    parser.add_argument('--save', action='store_true', help='Save model and results')
    args = parser.parse_args()

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run federated learning
    model, accuracies = federated_learning(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        local_epochs=args.local_epochs,
        dp_noise=args.dp_noise,
        enable_dp=not args.no_dp,
        batch_size=args.batch_size,
        lr=args.lr,
        save_results=args.save
    )

    # Save the model if requested
    if args.save:
        torch.save(model.state_dict(), 'federated_healthcare_model.pth')
        print("Model saved to federated_healthcare_model.pth")