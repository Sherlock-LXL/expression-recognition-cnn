import torch

from .model import NeuralNetwork
from .dataset import test_loader, test_data

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)

state_dict = torch.load(
    "./checkpoints/best_model_learning_rate_decay.pth",
    map_location=device
)
model.load_state_dict(state_dict)
model.eval()
print("Checkpoint loaded successfully.")

class_correct = [0] * 7
class_total = [0] * 7
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        for prediction, label in zip(pred, labels):
            class_total[label.item()] += 1
            if prediction == label:
                class_correct[label.item()] += 1

for i, class_name in enumerate(test_data.classes):
    accuracy = class_correct[i] / class_total[i]
    print(
        f"{class_name}: "
        f"{accuracy * 100:.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )
print(f"Overall accuracy: {correct / total * 100:.2f}%")
