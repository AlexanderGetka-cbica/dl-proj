import torch
from torch import nn, Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import PretrainedConfig, PreTrainedModel
from jaxtyping import Float
from tqdm import tqdm
from pandas import DataFrame
from einops import rearrange

import os
print(os.getcwd())
from shared.components import Linear, Bilinear

def _collator(transform=None):
    def inner(batch):
        xs, ys = [], []
        for img, label in batch:
            # Apply transform to each image individually if provided
            if transform is not None:
                img = transform(img)
            xs.append(img)
            ys.append(label)
        
        # Stack all images and labels
        # After stacking: x has shape [B, C, H, W]
        x = torch.stack(xs).float()
        y = torch.tensor(ys, dtype=torch.long)
        
        return x, y
    return inner

class Config(PretrainedConfig):
    def __init__(
        self,
        lr: float = 1e-3,
        wd: float = 0.5,
        epochs: int = 100,
        batch_size: int = 2048,
        d_hidden: int = 256,
        d_output: int = 10,
        bias: bool = False,
        seed: int = 42,
        **kwargs
    ):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.d_hidden = d_hidden
        self.d_output = d_output
        self.bias = bias
        super().__init__(**kwargs)

class Model(PreTrainedModel):
    """
    A 3-layer bilinear CNN model with input size 224x224.
    Instead of a standard non-linear activation function,
    we use the Bilinear layer as the 'activation'.
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        torch.manual_seed(config.seed)
        
        # We'll define a simple 3-layer CNN with downsampling.
        # After each Conv layer, we apply a Bilinear layer across channels.

        # Input: 224x224
        # After Conv1 (stride=2): 112x112
        # After Conv2 (stride=2): 56x56
        # After Conv3 (stride=2): 28x28
        # We'll pick channel dimensions somewhat arbitrarily:
        c1, c2, c3 = 64, 128, 256
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c1, kernel_size=3, stride=2, padding=1, bias=config.bias)
        self.bil1 = Bilinear(c1, c1, bias=config.bias)
        
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels=c2, kernel_size=3, stride=2, padding=1, bias=config.bias)
        self.bil2 = Bilinear(c2, c2, bias=config.bias)
        
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=3, stride=2, padding=1, bias=config.bias)
        self.bil3 = Bilinear(c3, c3, bias=config.bias)
        
        # After the last conv, feature map size is (B, c3, 28, 28)
        # Flatten and apply a final linear layer for classification.
        self.fc = Linear(c3*28*28, config.d_output, bias=config.bias)

        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = lambda y_hat, y: (y_hat.argmax(dim=-1) == y).float().mean()
    
    def forward(self, x: Float[Tensor, "batch 3 224 224"]) -> Float[Tensor, "batch d_output"]:
        # Conv1 + bilinear activation
        x = self.conv1(x)  # (B, c1, 112, 112)
        x = self.apply_bilinear(x, self.bil1)
        
        # Conv2 + bilinear activation
        x = self.conv2(x)  # (B, c2, 56, 56)
        x = self.apply_bilinear(x, self.bil2)
        
        # Conv3 + bilinear activation
        x = self.conv3(x)  # (B, c3, 28, 28)
        x = self.apply_bilinear(x, self.bil3)
        
        # Flatten and classify
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x
    
    def apply_bilinear(self, x, bilayer: Bilinear):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Reshape to apply bilinear layer as if it's a "channel-wise" activation
        x = rearrange(x, "b c h w -> (b h w) c")
        x = bilayer(x)  # (B*H*W, C)
        x = rearrange(x, "(b h w) c -> b c h w", b=B, h=H, w=W)
        return x

    @classmethod
    def from_config(cls, *args, **kwargs):
        return cls(Config(*args, **kwargs))

    @classmethod
    def from_pretrained(cls, path, *args, **kwargs):
        new = cls(Config(*args, **kwargs))
        new.load_state_dict(torch.load(path))
        return new
    
    def step(self, x, y):
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = self.accuracy(y_hat, y)
        return loss, accuracy
    
    def train(self, train, test, transform=None):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(True)
        
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        
        loader = DataLoader(train, batch_size=self.config.batch_size, shuffle=True, drop_last=True, collate_fn=_collator(transform))
        test_x, test_y = test.data, test.targets
        
        pbar = tqdm(range(self.config.epochs))
        history = []
        
        for _ in pbar:
            epoch = []
            for x, y in loader:
                outputs = self(x)
                loss = self.criterion(outputs, y)
                
                #compute accuracy
                _ , preds = torch.max(outputs, 1)
                acc = (preds == y).float().mean()
                
                epoch.append((loss.item(), acc.item()))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_loss, val_acc = self.eval().step(test_x, test_y)
            metrics = {
                "train/loss": sum(l for l, _ in epoch) / len(epoch),
                "train/acc": sum(a for _, a in epoch) / len(epoch),
                "val/loss": val_loss.item(),
                "val/acc": val_acc.item()
            }
            
            history.append(metrics)
            pbar.set_description(', '.join(f"{k}: {v:.3f}" for k, v in metrics.items()))
        
        torch.set_grad_enabled(False)
        return DataFrame.from_records(history, columns=['train/loss', 'train/acc', 'val/loss', 'val/acc'])

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transforms for CIFAR-10
# Convert the image to 224x224 and to a Tensor.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load CIFAR-10 training and test sets
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# CIFAR-10: 10 classes
num_classes = 10

# Create a wrapper for test dataset that makes test.x and test.y easily accessible
class DatasetWrapper:
    def __init__(self, dataset):
        xs = []
        ys = []
        for img, label in dataset:
            xs.append(img)
            ys.append(label)
        
        # Stack all images and labels into single tensors
        # Note: This loads the entire test set into memory (10,000 images for CIFAR-10).
        # For large datasets, you might want a different validation routine.
        self.x = torch.stack(xs)
        self.y = torch.tensor(ys, dtype=torch.long)

test_wrapper = DatasetWrapper(test_dataset)

# Instantiate the model from the provided code
model = Model.from_config(
    lr=1e-3,
    wd=0.5,
    epochs=5,
    batch_size=64,   # CIFAR-10 allows a smaller batch size due to smaller images (although we resize)
    d_hidden=256,
    d_output=num_classes,
    bias=False,
    seed=42
)

print(f"train dataset shape: {train_dataset.data.shape}")
print(f"test dataset shape: {test_dataset.data.shape}")


# Train the model
history = model.train(train_dataset, test_dataset)
print(history)

