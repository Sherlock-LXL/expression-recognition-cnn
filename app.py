import torch
from PIL import Image
from torchvision import transforms
import sys
from pathlib import Path

from src.model import NeuralNetwork
from src.config import CLASS_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NeuralNetwork().to(device)
state_dict = torch.load(
    "./checkpoints/best_model.pth",
    map_location=device,
)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
])

if len(sys.argv) < 2:
    print("Usage: py app.py <image_name>")
    sys.exit()
image_name = sys.argv[1]

image_path = Path("samples") / image_name
image = Image.open(image_path)
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

with torch.no_grad():
    logits = model(image)

probabilities = torch.softmax(logits, dim=1)
top_probs, top_indices = torch.topk(probabilities, k=3, dim=1)
top_probs = top_probs[0]
top_indices = top_indices[0]

for prob, index in zip(top_probs, top_indices):
    class_name = CLASS_NAMES[index.item()]
    print(f"{class_name}: {prob.item() * 100:.2f}%")