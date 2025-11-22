import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
import numpy as np

from preprocess_car_dd import CarDDDataset, CarDDNumpyDataset


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
            label = labels[0] - 1
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


# train + accuracy + confusion matrix + validation
def train(model, train_loader, val_loader, device, epochs=15):
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-4)

    for ep in range(1, epochs + 1):
        # train
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        for imgs, labels in train_loader:
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
        print(f"Epoch {ep} | Loss: {total_loss/len(train_loader):.4f} | Acc: {acc:.4f}")

        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)

                out = model(imgs)
                _, pred = torch.max(out, 1)

                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        print(f"Epoch {ep} | Val Acc: {val_acc:.4f}")

    # confusion matrix for train + val
    print("\nTrain Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nVal Confusion Matrix:")
    print(confusion_matrix(val_labels, val_preds))


# main
def main():
    images_dir = "../Dataset/train/"
    annotations_path = "../Dataset/train.json"

    # validation files you already have
    val_images_dir = "../Dataset/val"
    val_annotations_path = "../Dataset/val.json"

    # toggle to use preprocessed numpy bundles instead of raw images/JSON
    use_numpy_bundles = True
    train_bundle_dir = "../DatasetPreprocessed/train"
    val_bundle_dir = "../DatasetPreprocessed/val"

    img_size = 128
    batch_size = 8
    num_classes = 6

    if use_numpy_bundles:
        base = CarDDNumpyDataset(
            bundle_dir=train_bundle_dir,
            normalize=True,
            build_semantic=False
        )
        val_base = CarDDNumpyDataset(
            bundle_dir=val_bundle_dir,
            normalize=True,
            build_semantic=False
        )
    else:
        # training dataset
        base = CarDDDataset(
            images_dir=images_dir,
            annotations_path=annotations_path,
            target_size=img_size,
            build_semantic=False,
            normalize=True
        )

        # validation dataset
        val_base = CarDDDataset(
            images_dir=val_images_dir,
            annotations_path=val_annotations_path,
            target_size=img_size,
            build_semantic=False,
            normalize=True
        )

    cls_ds = CarDD_Cls(base)
    val_ds = CarDD_Cls(val_base)

    # compute class frequencies from train only
    labels_list = []
    for i in range(len(cls_ds)):
        _, lbl = cls_ds[i]
        labels_list.append(int(lbl))

    labels_np = np.array(labels_list)
    class_counts = np.bincount(labels_np, minlength=6)
    class_weights = 1.0 / (class_counts + 1e-6)

    sample_weights = class_weights[labels_np]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # same epoch size
        replacement=True
    )

    # loader for train (weighted) + val (normal)
    train_loader = DataLoader(cls_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = BasicCNN(num_classes=num_classes, img_size=img_size).to(device)

    train(model, train_loader, val_loader, device, epochs=15)

    torch.save(model.state_dict(), "car_dd_cnn.pth")
    print("model saved")


if __name__ == "__main__":
    main()
