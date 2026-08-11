import torch

from .model import NeuralNetwork
from .dataset import test_loader, test_data

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_paths = [
    "./checkpoints/best_model_res_1.pth",
    "./checkpoints/best_model_res_2.pth",
    "./checkpoints/best_model_res_3.pth",
]

models = []

for path in checkpoint_paths:
    model = NeuralNetwork().to(device)
    state_dict = torch.load(
        path,
        map_location=device
    )
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
print(f"{len(models)} checkpoints loaded successfully.")

class_correct = [0] * 7
class_total = [0] * 7
correct = 0
total = 0

with torch.inference_mode():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        probabilities = torch.zeros(
            images.size(0),
            7,
            device=device
        )

        for model in models:
            logits = model(images)
            probs = torch.softmax(
                logits,
                dim=1
            )
            probabilities += probs

        probabilities /= len(models)
        pred = probabilities.argmax(dim=1)
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

print(
    f"Overall accuracy: "
    f"{correct / total * 100:.2f}%"
)