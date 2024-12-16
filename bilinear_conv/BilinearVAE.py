from conv_CBilA import BilinearCNN, Bilinear
from einops import *
import torch
import torch.nn as nn
from decompositions import decompose_bilinear_layer
import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import wandb
from datetime import datetime
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class Encoder(torch.nn.Module):
    def __init__(self, in_channels = 3, base_channels=64, latent_dim=128, bias=False):
        super().__init__()
        # A single convolution layer (you could add more or use residual blocks here if desired)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, padding=1, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear', a=0.1)
        self.bn = nn.BatchNorm2d(base_channels)
        # A max pooling step to reduce spatial dimensions, e.g. from 32x32 to 16x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc_mu = nn.Linear(base_channels * 16 * 16, 128)
        self.fc_logvar = nn.Linear(base_channels * 16 * 16, 128)
    
    def forward(self, x):
        print(f"Encoder input shape: {x.shape}")
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        print(f"Encoder output mu shape: {mu.shape}")
        print(f"Encoder output logvar shape: {logvar.shape}")
        return mu, logvar

class Classifier(torch.nn.Module):
    def __init__(self, hidden_dim, num_classes=10, bias=False, gate=None):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Here the bilinear layer acts as the "activation function" replacement
        self.bilinear = Bilinear(hidden_dim, hidden_dim, bias=bias, gate=gate)
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=bias)
        
    def forward(self, x):
        print(f"Classifier input shape: {x.shape}")
        x = self.fc1(x)
        x = self.bilinear(x)
        x = self.fc2(x)
        print(f"Classifier output shape: {x.shape}")
        return x


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, base_channels=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels // 2, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Scale output to [-1, 1]
        )

    def forward(self, x):
        print(f"Decoder input shape: {x.shape}")
        x = self.fc(x)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.deconv(x)
        print(f"Decoder output shape: {x.shape}")
        return x

class BilinearVAE(torch.nn.Module):
    def __init__(self, encoder, classifier, decoder, num_samples=5):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.decoder = decoder
        self.num_samples = num_samples

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn(self.num_samples, *mu.size(), device=mu.device)  # Generate multiple samples
        return mu.unsqueeze(0) + eps * std.unsqueeze(0)  # Shape: [num_samples, batch_size, latent_dim]

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z_samples = self.reparameterize(mu, logvar)  # Shape: [num_samples, batch_size, latent_dim]
        print(f"z_samples shape: {z_samples.shape}")
        reconstructions = torch.stack([self.decoder(z) for z in z_samples], dim=0)  # Reconstruction per sample
        classifications = torch.stack([self.classifier(z) for z in z_samples], dim=0)  # Class predictions per sample
        return reconstructions, classifications, mu, logvar


# Loss Function
def vae_loss(reconstructions, x, classifications, labels, mu, logvar, beta=1.0):
    print(f"Reconstructions shape: {reconstructions.shape}")
    print(f"Classifications shape: {classifications.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")

    # Reconstruction loss: Average over multiple samples
    recon_loss = torch.mean(
        torch.stack([nn.MSELoss(reduction="sum")(recon, x) for recon in reconstructions], dim=0)
    )
    # Classification loss: Cross-entropy averaged across samples
    print("Class prediction shapes:")
    print([class_preds.shape for class_preds in classifications])
    class_loss = torch.mean(
        torch.stack([nn.CrossEntropyLoss()(class_preds, labels) for class_preds in classifications], dim=0)
    )
    # KL divergence: Regularize the latent space
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld_loss + class_loss

def train_model(config, train_dataset, test_dataset):
    pass


#model = BilinearCNN(base_channels=64, num_classes=10, bias=False, gate='none')
#model.load_state_dict("trained_bilinear_model.pth")
#model = model.to(device)
#model.eval()

# Example Initialization
encoder = Encoder(in_channels=3, base_channels=64)
classifier = Classifier(hidden_dim=128, num_classes=10, gate=None)
decoder = Decoder(latent_dim=128)
vae = BilinearVAE(encoder, classifier, decoder, num_samples=5).to(device)

run = wandb.init(
    project="bilinear_cnn_vae",  # Replace with your project name
    name="decoder_training",     # Optional: Name for this run
    config={
        "latent_dim": 128,
        "base_channels": 64,
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 20
    }
)



# Training Loop
def train_vae(model, train_loader, val_loader, optimizer, epochs=10, beta=1.0):
    model.train()
    # Metrics to track over epochs
    train_recon_losses, train_class_losses, train_kld_losses, train_total_losses = [], [], [], []
    val_recon_losses, val_kld_losses, val_class_losses, val_total_losses = [], [], [], []
    train_errors, val_errors = [], []
    for epoch in range(epochs):
        # Initialize running totals
        total_recon_loss = 0.0
        total_class_loss = 0.0
        total_kld_loss = 0.0
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, labels in train_loader:
            if x.size(0) != 128:
                # skip to avoid bug with batch sizes, don't have time to fix
                continue
            x, labels = x.to(device), labels.to(device)
            print(f"x shape: {x.shape}")
            print(f"labels shape: {labels.shape}")
            optimizer.zero_grad()

            # Forward pass
            reconstructions, classifications, mu, logvar = model(x)
            print(f"reconstructions shape: {reconstructions.shape}")
            print(f"classifications shape: {classifications.shape}")
            print(f"mu shape: {mu.shape}")
            print(f"logvar shape: {logvar.shape}")
            # Compute each component of the loss
            recon_loss = torch.mean(
                torch.stack([nn.MSELoss(reduction="sum")(recon, x) for recon in reconstructions], dim=0)
            )
            print(f"class_preds shapes:")
            print([class_preds.shape for class_preds in classifications])
            class_loss = torch.mean(
                torch.stack([nn.CrossEntropyLoss()(class_preds, labels) for class_preds in classifications], dim=0)
            )
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            total_batch_loss = recon_loss + beta * kld_loss + class_loss

            # Backward pass and optimization
            total_batch_loss.backward()
            optimizer.step()

            # Track losses and classification accuracy
            total_recon_loss += recon_loss.item()
            total_class_loss += class_loss.item()
            total_kld_loss += kld_loss.item()
            total_loss += total_batch_loss.item()

            # Compute classification error (using the first sample for simplicity)
            predictions = torch.argmax(classifications[0], dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        # Print training statistics for the epoch
        train_class_error = 1 - (total_correct / total_samples)
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader):.4f}, "
            f"Recon Loss: {total_recon_loss / len(train_loader):.4f}, "
            f"Class Loss: {total_class_loss / len(train_loader):.4f}, "
            f"KLD Loss: {total_kld_loss / len(train_loader):.4f}, "
            f"Class Error: {train_class_error:.4f}"
        )

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_recon_loss = 0.0
            val_kld_loss = 0.0
            val_class_loss = 0.0
            val_total_loss = 0.0
            val_correct = 0
            val_samples = 0


            for x_val, labels_val in val_loader:
                x_val, labels_val = x_val.to(device), labels_val.to(device)
                if x_val.size(0) != 128:
                    # skip to avoid bug, don't have time to fix
                    continue
                reconstructions, classifications, mu, logvar = model(x_val)

                # Reconstruction loss
                val_recon_loss += torch.mean(
                    torch.stack([nn.MSELoss(reduction="sum")(recon, x_val) for recon in reconstructions], dim=0)
                ).item()

                print(f"class_preds sizes:")
                print([class_preds.shape for class_preds in classifications])
                print(f"labels_val shape: {labels_val.shape}")
                # Classification loss
                val_class_loss += torch.mean(
                    torch.stack([nn.CrossEntropyLoss()(class_preds, labels_val) for class_preds in classifications], dim=0)
                ).item()
                
                # KLD loss
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                val_kld_loss += kld_loss.item()

                val_loss = val_recon_loss + beta * val_kld_loss + val_class_loss
                val_total_loss += val_loss

                # Classification error
                val_predictions = torch.argmax(classifications[0], dim=1)
                val_correct += (val_predictions == labels_val).sum().item()
                val_samples += labels_val.size(0)

            val_class_error = 1 - (val_correct / val_samples)
            print(
                f"Validation Recon Loss: {val_recon_loss / len(val_loader):.4f}, "
                f"Validation KLD Loss: {val_kld_loss / len(val_loader):.4f}, "
                f"Validation Class Loss: {total_class_loss / len(val_loader):.4f}, "
                f"Validation Class Error: {val_class_error:.4f}"
            )

        model.train()
        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "train_error": train_class_error,
            "val_loss": val_total_loss / len(val_loader),
            "val_error": val_class_error
        })
    # Plot metrics
    epochs_range = range(1, epochs + 1)

    # Plot training and validation errors
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, train_errors, label="Train Classification Error")
    # plt.plot(epochs_range, val_errors, label="Validation Classification Error")
    # plt.xlabel("Epochs")
    # plt.ylabel("Error")
    # plt.title("Classification Error vs Epochs")
    # plt.legend()
    # plt.show()

    # # Plot training and validation losses
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, train_total_losses, label="Train Total Loss")
    # plt.plot(epochs_range, val_total_losses, label="Validation Total Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Train and Validation Total Loss vs Epochs")
    # plt.legend()
    # plt.show()

    # # Plot KL vs Reconstruction Loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, train_kld_losses, label="KL Divergence Loss")
    # plt.plot(epochs_range, train_recon_losses, label="Reconstruction Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("(Training) KL Divergence and Reconstruction Loss vs Epochs")
    # plt.legend()
    # plt.show()

    # # Plot KL vs Reconstruction Loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, val_kld_losses, label="KL Divergence Loss")
    # plt.plot(epochs_range, val_recon_losses, label="Reconstruction Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("(Validation) KL Divergence and Reconstruction Loss vs Epochs")
    # plt.legend()
    # plt.show()

    # # Plot Classification Loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(epochs_range, train_class_losses, label="Train Classification Loss")
    # plt.plot(epochs_range, val_class_losses, label="Validation Classification Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Classification Loss vs Epochs")
    # plt.legend()
    # plt.show()





# Example Training (Assuming train_loader exists)

config = {
"learning_rate": 3e-4,
"weight_decay": 5e-4, # change to 1e-3 after this run
"batch_size": 128,
"num_blocks": 3,
"base_channels": 64,
"grad_clip": 0.5,
"epochs": 100
}

# autoaugment
transform = transforms.Compose([
transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
transforms.ToTensor(),
transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
# Other augmentations I could try:
#Augmix
#Kernel Filtering
#Discrete Cosine Transform
# transform = transforms.Compose([
#     transforms.RandomResizedCrop(32),
#     AugMix(severity=3, width=3, depth=2, alpha=1.),
#     KernelFilter(kernel_size=3),
#     DCTAugmentation(factor=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

full_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])
)

train_loader = DataLoader(
       full_train_dataset,
       batch_size=config['batch_size'],
       shuffle=True,
       num_workers=4,
       pin_memory=True
   )
   
test_loader = DataLoader(
       test_dataset,
       batch_size=config['batch_size'], 
       shuffle=False,
       num_workers=4,
       pin_memory=True
   )

optimizer = AdamW(
    vae.parameters(),
    lr=config['learning_rate'],
    weight_decay=config['weight_decay']
)

steps_per_epoch = len(train_loader)
scheduler = OneCycleLR(
    optimizer,
    max_lr=config['learning_rate'],
    epochs=config['epochs'],
    steps_per_epoch=steps_per_epoch,
    pct_start=0.15,
    div_factor=10,
    final_div_factor=1e4
)

if __name__ == "__main__":
    train_vae(vae, train_loader, test_loader, optimizer, epochs=10)

    torch.save(vae.state_dict(), "trained_vae.pth")
    torch.save(encoder.state_dict(), "trained_encoder.pth")
    torch.save(classifier.state_dict(), "trained_classifier.pth")
    torch.save(decoder.state_dict(), "trained_decoder.pth")


