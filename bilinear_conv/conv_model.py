import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import RandomErasing, AutoAugment, AutoAugmentPolicy
from tqdm import tqdm
from itertools import product
import numpy as np
import pandas as pd
from pathlib import Path
import json
import wandb
from datetime import datetime

# Constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SAVE_DIR = Path("model_checkpoints")
SAVE_DIR.mkdir(exist_ok=True)

# Grid Search Parameters
# param_grid = {
#     'learning_rate': [3e-4, 1e-3],    # Increase LR since current is too slow
#     'weight_decay': [5e-4, 1e-3],     # Increase to help regularization
#     'batch_size': [128],              # Larger batch for stability
#     'num_blocks': [3, 4],             # Keep testing both
#     'base_channels': [64],            # Increase capacity
#     'grad_clip': [1.0],               # Current 0.5 might be too restrictive
#     'epochs': [100]                   # Train longer since still improving
# }

channel_configs = {
    3: lambda base: [base, base*2, base*4],
    4: lambda base: [base, base*2, base*4, base*8],
}

class BConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv_u = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        self.conv_v = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=bias)
        nn.init.kaiming_normal_(self.conv_u.weight, nonlinearity='linear', a=0.1)
        nn.init.kaiming_normal_(self.conv_v.weight, nonlinearity='linear', a=0.1)
    
    def forward(self, x):
        return self.conv_u(x) * self.conv_v(x)

class BilinearCNN(nn.Module):
    def __init__(self, num_blocks, base_channels, num_classes=10, bias=False):
        super().__init__()
        self.channels = channel_configs[num_blocks](base_channels)
        
        # Create blocks dynamically
        self.blocks = nn.ModuleList()
        in_channels = 3
        
        for out_channels in self.channels:
            block = nn.Sequential(
                BConv2d(in_channels, out_channels, bias=bias),
                nn.BatchNorm2d(out_channels)
            )
            self.blocks.append(block)
            in_channels = out_channels
        
        # Calculate final feature size
        # For CIFAR-10: Starting size is 32x32
        # Each BConv2d uses stride=1, so spatial dimensions remain the same
        final_size = 32  # Since we're not reducing spatial dimensions in conv layers
        final_channels = self.channels[-1]
        
        # Print dimensions for debugging
        print(f"Final channels: {final_channels}")
        print(f"Final spatial size: {final_size}")
        print(f"Final features: {final_channels * final_size * final_size}")
        
        self.fc = nn.Linear(final_channels * final_size * final_size, 
                           num_classes, bias=bias)

    def forward(self, x):
        # Print shape at each step for debugging
        # print(f"Input shape: {x.shape}")
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            # print(f"After block {i+1}: {x.shape}")
        
        x = x.flatten(start_dim=1)
        # print(f"After flatten: {x.shape}")
        
        x = self.fc(x)
        # print(f"Final output: {x.shape}")
        return x

def train_epoch(model, train_loader, optimizer, scheduler, grad_clip, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Training')
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'loss': f'{total_loss/(batch_idx+1):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    
    return test_loss, accuracy

def train_model(config, train_dataset, test_dataset):
   run = wandb.init(project="bilinear-cnn-gridsearch", config=config)
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   save_dir = Path("checkpoints")
   save_dir.mkdir(exist_ok=True)
   
   print(f"Using device: {device}")
   
   train_loader = DataLoader(
       train_dataset,
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
   
   model = BilinearCNN(
       num_blocks=config['num_blocks'],
       base_channels=config['base_channels']
   ).to(device)
   
   optimizer = AdamW(
       model.parameters(),
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
   
   # Training loop
   best_acc = 0.0
   for epoch in range(config['epochs']):
       # Training
       train_loss, train_acc = train_epoch(
           model, train_loader, optimizer, scheduler, config['grad_clip'], device
       )
       
       # Validation every 5 epochs or on final epoch
       if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
           test_loss, test_acc = evaluate(model, test_loader, device)
           
           print(f"\nEpoch {epoch+1}/{config['epochs']}")
           print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
           print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")
           
           # Log to wandb
           wandb.log({
               "epoch": epoch,
               "train_loss": train_loss,
               "train_acc": train_acc,
               "test_loss": test_loss, 
               "test_acc": test_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
           # Save best model
           if test_acc > best_acc:
               best_acc = test_acc
               torch.save({
                   'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'test_acc': test_acc,
                   'config': config
               }, save_dir / f"model_best_{timestamp}.pt")
       else:
           # Log only training metrics for other epochs
           wandb.log({
               "epoch": epoch,
               "train_loss": train_loss, 
               "train_acc": train_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
           print(f"Epoch {epoch+1}/{config['epochs']}")
           print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

   # Save final model
   torch.save({
       'epoch': config['epochs'],
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'test_acc': test_acc,
       'config': config
   }, save_dir / f"model_final_{timestamp}.pt")

   wandb.finish()
   return best_acc


def continue_training(model_path, train_dataset, test_dataset, additional_epochs=100):
   # Fixed config
   
   wandb.init(project="bilinear-cnn-gridsearch", 
           name="continued_training",
           config={
               "learning_rate": 3e-4,
               "weight_decay": 5e-4,
               "batch_size": 128,
               "num_blocks": 3,
               "base_channels": 64,
               "grad_clip": 0.5,
               "epochs": 50,
               "continued_from": model_path
           })
   
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   # explicitly define the best hyperparameters
   config = {
       "learning_rate": 3e-4,
       "weight_decay": 5e-4,
       "batch_size": 128,
       "num_blocks": 3,
       "base_channels": 64,
       "grad_clip": 0.5,
       "epochs": additional_epochs
   }
   
   # Load and inspect checkpoint
   checkpoint = torch.load(model_path)
   print("Keys in checkpoint:", checkpoint.keys())
   
   # Initialize model and load state
   model = BilinearCNN(
       num_blocks=config['num_blocks'],
       base_channels=config['base_channels']
   ).to(device)
   
   # Load state dict directly if that's how it was saved
   if isinstance(checkpoint, dict):
       if 'model_state_dict' in checkpoint:
           model.load_state_dict(checkpoint['model_state_dict'])
       else:
           model.load_state_dict(checkpoint)
   else:
       model.load_state_dict(checkpoint)

   train_loader = DataLoader(
       train_dataset,
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
       model.parameters(),
       lr=config['learning_rate'],
       weight_decay=config['weight_decay']
   )
   
   steps_per_epoch = len(train_loader)
   scheduler = OneCycleLR(
       optimizer,
       max_lr=config['learning_rate'],
       epochs=config['epochs'],
       steps_per_epoch=steps_per_epoch,
       pct_start=0.17,
       div_factor=10,
       final_div_factor=1e4
   )

   # Continue training
   best_acc = checkpoint['test_acc']
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
   for epoch in range(config['epochs']):
       train_loss, train_acc = train_epoch(
           model, train_loader, optimizer, scheduler, config['grad_clip'], device
       )
       
       if (epoch + 1) % 5 == 0 or epoch == config['epochs'] - 1:
           test_loss, test_acc = evaluate(model, test_loader, device)
           
           print(f"\nEpoch {epoch+1}/{config['epochs']}")
           print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
           print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}\n")
           
           wandb.log({
               "epoch": epoch + checkpoint['epoch'],  # Continue epoch counting
               "train_loss": train_loss,
               "train_acc": train_acc,
               "test_loss": test_loss,
               "test_acc": test_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
           if test_acc > best_acc:
               best_acc = test_acc
               torch.save({
                   'epoch': epoch + checkpoint['epoch'],
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'test_acc': test_acc,
                   'train_acc': train_acc,
                   'config': config,
               }, f"continued_ep_400_{timestamp}.pt")
       else:
           wandb.log({
               "epoch": epoch + checkpoint['epoch'],
               "train_loss": train_loss,
               "train_acc": train_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
           print(f"Epoch {epoch+1}/{config['epochs']}")
           print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

   wandb.finish()
   return best_acc  
    

def main():
    
#     transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
#     transforms.RandomGrayscale(p=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
#     RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random')
# ])
    
    # Try this next morning, if test accuracy didn't exceed 80%
    # autoaugment
    transform = transforms.Compose([
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])
    
    #Augmix
    #Kernel Filtering
    #Discrete Cosine Transform

    full_dataset = datasets.CIFAR10(
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

    continue_training("continued_ep_300_20241215_071749.pt", full_dataset, test_dataset, additional_epochs=150)

if __name__ == "__main__":
    main()
    
# def main():
#     # Fixed configuration
#     config = {
#         "learning_rate": 3e-4,
#         "weight_decay": 5e-4, # change to 1e-3 after this run
#         "batch_size": 128,
#         "num_blocks": 3,
#         "base_channels": 64,
#         "grad_clip": 0.5,
#         "epochs": 100
#     }

#     # Data augmentation
#     transform = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomAffine(
#             degrees=15,
#             shear=10,
#             translate=(0.1, 0.1)
#         ),
#         transforms.ColorJitter(
#             brightness=0.2,
#             contrast=0.2,
#             saturation=0.2
#         ),
#         transforms.RandomAdjustSharpness(sharpness_factor=1.5),
#         transforms.RandomPosterize(bits=6),
#         transforms.RandomSolarize(threshold=192),
#         transforms.RandomEqualize(),
#         transforms.ToTensor(),
#         transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
#         ])
    
#     # Load CIFAR-10
#     full_dataset = datasets.CIFAR10(
#         root="./data",
#         train=True,
#         download=True,
#         transform=transform
#     )
    
#     test_dataset = datasets.CIFAR10(
#         root="./data",
#         train=False,
#         download=True,
#         transform=transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
#         ])
#     )
    
#     # Create 50% subset
#     # subset_size = int(0.5 * len(full_dataset))
#     torch.manual_seed(42)
#     indices = torch.randperm(len(full_dataset))#[:subset_size]
#     train_dataset = Subset(full_dataset, indices)
    
#     print(f"Training dataset size: {len(train_dataset)}")
#     print(f"Test dataset size: {len(test_dataset)}")
    
#     print("\nTraining with configuration:")
#     print(json.dumps(config, indent=2))
    
#     best_acc = train_model(config, train_dataset, test_dataset)
#     print(f"\nBest Test Accuracy: {best_acc:.4f}")

# if __name__ == "__main__":
#     main()
