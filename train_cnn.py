import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np

from preprocess_car_dd import CarDDDataset


# simple classification wrapper
class CarDD_Cls(Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        rec = self.base[idx]
        img = rec["image"]
        labels = rec["target"]["labels"]

        # first label or 0
        if len(labels) > 0:
            label = labels[0]
        else:
            label = torch.tensor(0, dtype=torch.long)

        return img, label


# basic cnn
class BasicCNN(nn.Module):
    def __init__(self, num_classes=6, img_size=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        s = img_size // 8  # 512 -> 64

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * s * s, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# train + accuracy + confusion matrix
def train(model, loader, device, epochs=5):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item()

            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        acc = correct / total
        print(f"Epoch {ep} | Loss: {total_loss/len(loader):.4f} | Acc: {acc:.4f}")

    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)


# main
def main():
    images_dir = "./car_dd_images"
    annotations_path = "./instances_train2017.json"
    img_size = 512
    batch_size = 8
    num_classes = 6

    base = CarDDDataset(
        images_dir=images_dir,
        annotations_path=annotations_path,
        target_size=img_size,
        build_semantic=False,
        normalize=True
    )

    cls_ds = CarDD_Cls(base)
    loader = DataLoader(cls_ds, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = BasicCNN(num_classes=num_classes, img_size=img_size).to(device)

    train(model, loader, device, epochs=5)

    torch.save(model.state_dict(), "car_dd_cnn.pth")
    print("model saved")


if __name__ == "__main__":
    main()
