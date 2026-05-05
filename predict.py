import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
import os

from model import get_model


# 🔹 Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔹 Load class names (same order as training)
def get_classes(data_dir):
    classes = sorted(os.listdir(data_dir))
    return classes


# 🔹 Image transform (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# 🔹 Predict function
def predict_image(image_path, model, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return classes[pred.item()]


# 🔹 Main
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]

    data_dir = "PokemonData"
    classes = get_classes(data_dir)

    model = get_model(len(classes), pretrained=False, freeze=False)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    result = predict_image(image_path, model, classes)

    print("\nPrediction:", result)