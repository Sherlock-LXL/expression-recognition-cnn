import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import SEED, BATCH_SIZE

train_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(
    "./data/train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    "./data/train",
    transform=val_transform
)

test_data = datasets.ImageFolder(
    "./data/test",
    transform=test_transform
)

total_size = len(train_data)
val_size = int(total_size / 10)
train_size = total_size - val_size

generator = torch.Generator().manual_seed(SEED)
train_split, val_split = torch.utils.data.random_split(
    train_data,
    [train_size, val_size],
    generator=generator
)

train_set = Subset(train_data, train_split.indices)
val_set = Subset(val_data, val_split.indices)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)