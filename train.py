import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
from dataset import RoadSceneDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 28  # 26 IDD classes + 1 pothole

def get_dataloaders(data_root, batch_size=4):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    train_ds = RoadSceneDataset(data_root, split="train", transform=transform)
    val_ds = RoadSceneDataset(data_root, split="val", transform=transform)

    print(f"✅ Found {len(train_ds)} training samples, {len(val_ds)} validation samples")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, val_loader

def get_model(num_classes):
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    model = models.segmentation.deeplabv3_resnet50(weights=weights)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model.to(DEVICE)

def train():
    print("🚀 Initializing training...")
    model = get_model(NUM_CLASSES)
    train_loader, val_loader = get_dataloaders("data/processed", batch_size=4)

    criterion = nn.CrossEntropyLoss(ignore_index=255)  # ✅ Fix for void labels
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, 11):
        model.train()
        total_loss = 0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(imgs)['out']

            # Resize masks to match output shape
            masks = nn.functional.interpolate(
                masks.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest"
            ).squeeze(1).long()

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/10] 🔁 Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "deeplabv3.pth")
    print("✅ Training complete — model saved as deeplabv3.pth")

if __name__ == "__main__":
    train()