import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# car-dd dataset
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
            nn.MaxPool2d(2),  # 512 -> 256

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 256 -> 128

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128 -> 64
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


# train loop
def train(model, loader, device, epochs=5):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            opt.zero_grad()
            out = model(imgs)
            loss = crit(out, labels)
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"Epoch {ep}: {total / len(loader):.4f}")


# main
def main():
    images_dir = "./car_dd_images"
    annotations_path = "./instances_train2017.json"
    img_size = 512
    batch_size = 8
    num_classes = 6  # car-dd has 6 damage classes

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
