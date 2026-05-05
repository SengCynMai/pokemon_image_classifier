import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import get_model

from sklearn.metrics import accuracy_score, precision_score, recall_score

print("START TRAINING")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# since you run from Assignment6 folder
data_dir = "PokemonData"

train_loader, val_loader, test_loader, classes = get_dataloaders(data_dir)

print("Number of classes:", len(classes))
print("Train batches:", len(train_loader))


# 🔹 Evaluation function
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')

    return acc, prec, rec


# 🔹 Model (baseline experiment)
model = get_model(len(classes), pretrained=True, freeze=False)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)


# 🔹 Training loop
for epoch in range(10):
    print(f"\nStarting Epoch {epoch+1}")

    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# 🔹 Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved!")


# 🔹 Evaluate
acc, prec, rec = evaluate(model, test_loader, device)

print("\nFinal Results:")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")