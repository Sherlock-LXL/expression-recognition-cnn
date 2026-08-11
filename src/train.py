import torch
from torch import nn
import os

from .dataset import train_loader, val_loader
from .model import NeuralNetwork
from .config import LEARNING_RATE, NUM_EPOCHS

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    return average_loss

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    average_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return average_loss, accuracy

print("Device:", device)
print("Model device:", next(model.parameters()).device)

best_val_accuracy = 0.0
os.makedirs("./checkpoints", exist_ok=True)

scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=[10],
    gamma=0.3
)

for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss, val_accuracy = validate(model, val_loader, loss_fn, device)
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            "./checkpoints/best_model.pth"
        )
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}\nTrain loss: {train_loss}\nVal loss: {val_loss}\nVal accuracy: {val_accuracy * 100}%\n")
    scheduler.step()
