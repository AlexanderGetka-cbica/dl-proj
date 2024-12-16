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

# Assuming CIFAR10_MEAN and CIFAR10_STD are defined elsewhere
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

class AugMix(object):
    def __init__(self, severity=3, width=3, depth=-1, alpha=1.):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha
        self.augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]

    def __call__(self, img):
        mix = img.copy()
        for _ in range(self.width):
            aug_img = img.copy()
            depth = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                aug_img = op(aug_img)
            mix = Image.blend(mix, aug_img, self.alpha)
        return mix


class KernelFilter(object):
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, img):
        return F.gaussian_blur(img, self.kernel_size)

class DCTAugmentation(object):
    def __init__(self, factor=0.1):
        self.factor = factor

    def __call__(self, img):
        img_np = np.array(img)
        dct_result = dct(dct(img_np.transpose(2, 0, 1), axis=1, norm='ortho'), axis=2, norm='ortho')
        dct_result += np.random.normal(0, self.factor * np.abs(dct_result), dct_result.shape)
        img_reconstructed = idct(idct(dct_result, axis=2, norm='ortho'), axis=1, norm='ortho').transpose(1, 2, 0)
        return Image.fromarray(np.uint8(np.clip(img_reconstructed, 0, 255)))


# Here we use your Bilinear and Linear classes from your snippet:
class Bilinear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Tensor) -> Tensor:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right
    
    @property
    def w_l(self):
        return self.weight.chunk(2, dim=0)[0]
    
    @property
    def w_r(self):
        return self.weight.chunk(2, dim=0)[1]

class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias=False, gate=None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU(), None: nn.Identity()}[gate]
    
    def forward(self, x: Tensor) -> Tensor:
        return self.gate(super().forward(x))

# A simplified BilinearCNN that:
# 1. Does a single (or a few) convolution(s) + max pool
# 2. Replaces the activation before the fully connected layer with a Bilinear layer

class BilinearCNN(nn.Module):
    def __init__(self, base_channels=64, num_classes=10, bias=False, gate=None):
        super().__init__()
        
        # A single convolution layer (you could add more or use residual blocks here if desired)
        self.conv = nn.Conv2d(in_channels=3, out_channels=base_channels, kernel_size=3, padding=1, bias=bias)
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity='linear', a=0.1)
        self.bn = nn.BatchNorm2d(base_channels)

        # A max pooling step to reduce spatial dimensions, e.g. from 32x32 to 16x16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # After pooling: final spatial size is 16x16 for CIFAR-10 input
        # Flatten and apply a bilinear layer as a "nonlinear activation" replacement
        final_size = 16
        final_channels = base_channels
        hidden_dim = final_channels * final_size * final_size
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        # Here the bilinear layer acts as the "activation function" replacement
        self.bilinear = Bilinear(hidden_dim, hidden_dim, bias=bias, gate=gate)

        # Finally, a linear classifier on top
        self.fc2 = nn.Linear(hidden_dim, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.bilinear(x)
        x = self.fc2(x)
        return x

###################################################################
# Example training code (remains very similar to your original)
###################################################################

def train_epoch(model, train_loader, optimizer, scheduler, grad_clip, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(train_loader, desc='Training')
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
   # Initialize wandb with a new project name and optionally a run name
   run = wandb.init(project="bilinear_cnn_activation", config=config, name="activation_replacement_experiment")
   
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
       base_channels=config['base_channels'],
       num_classes=10,
       bias=False,
       gate=None
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
   
   best_acc = 0.0
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
               "epoch": epoch,
               "train_loss": train_loss,
               "train_acc": train_acc,
               "test_loss": test_loss, 
               "test_acc": test_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
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
           wandb.log({
               "epoch": epoch,
               "train_loss": train_loss, 
               "train_acc": train_acc,
               "learning_rate": scheduler.get_last_lr()[0]
           })
           
           print(f"Epoch {epoch+1}/{config['epochs']}")
           print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

   torch.save({
       'epoch': config['epochs'],
       'model_state_dict': model.state_dict(),
       'optimizer_state_dict': optimizer.state_dict(),
       'test_acc': test_acc,
       'config': config
   }, save_dir / f"model_final_{timestamp}.pt")

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

    best_acc = train_model(config, full_train_dataset, test_dataset)

if __name__ == "__main__":
    main()
    
