import argparse
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as T

from preprocess_car_dd import CarDDDataset, CarDDNumpyDataset
from train_cnn import CarDD_Cls


def get_resnet_weights(model_name: str, use_pretrained: bool):
    """Return torchvision weights enum (if available) and the corresponding mean/std."""
    default_norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if not use_pretrained:
        return None, default_norm

    weight_map = {
        "resnet18": models.ResNet18_Weights if hasattr(models, "ResNet18_Weights") else None,
        "resnet34": models.ResNet34_Weights if hasattr(models, "ResNet34_Weights") else None,
        "resnet50": models.ResNet50_Weights if hasattr(models, "ResNet50_Weights") else None,
        "resnet101": models.ResNet101_Weights if hasattr(models, "ResNet101_Weights") else None,
        "resnet152": models.ResNet152_Weights if hasattr(models, "ResNet152_Weights") else None,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights if hasattr(models, "Wide_ResNet50_2_Weights") else None,
        "wide_resnet101_2": models.Wide_ResNet101_2_Weights if hasattr(models, "Wide_ResNet101_2_Weights") else None,
    }

    weights_enum = weight_map.get(model_name)
    if weights_enum is None:
        return None, default_norm

    weights = weights_enum.DEFAULT
    mean = weights.meta.get("mean", default_norm[0])
    std = weights.meta.get("std", default_norm[1])
    return weights, (mean, std)


def build_resnet(model_name: str, num_classes: int, pretrained: bool, freeze_backbone: bool):
    weights, _ = get_resnet_weights(model_name, pretrained)

    model_ctor = getattr(models, model_name)
    if weights is not None:
        model = model_ctor(weights=weights)
    else:
        # Fallback to deprecated arg for older torchvision versions
        model = model_ctor(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    return model


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size
        correct += (preds == labels).sum().item()
        all_labels.extend(labels.detach().cpu().tolist())
        all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, all_labels, all_preds


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, List[int], List[int]]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            correct += (preds == labels).sum().item()
            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, all_labels, all_preds


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    class_weights: Optional[torch.Tensor] = None,
):
    weight = class_weights.to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_val_acc = 0.0
    best_state = None
    last_train_labels: List[int] = []
    last_train_preds: List[int] = []
    last_val_labels: List[int] = []
    last_val_preds: List[int] = []
    best_val_labels: List[int] = []
    best_val_preds: List[int] = []
    best_train_labels: List[int] = []
    best_train_preds: List[int] = []

    for ep in range(1, epochs + 1):
        train_loss, train_acc, last_train_labels, last_train_preds = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, last_val_labels, last_val_preds = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_val_labels = last_val_labels.copy()
            best_val_preds = last_val_preds.copy()
            best_train_labels = last_train_labels.copy()
            best_train_preds = last_train_preds.copy()

        print(
            f"Epoch {ep:02d} | "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
            f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    # Prefer metrics from the best checkpoint; fall back to last if none
    train_labels_out = best_train_labels if best_train_labels else last_train_labels
    train_preds_out = best_train_preds if best_train_preds else last_train_preds
    val_labels_out = best_val_labels if best_val_labels else last_val_labels
    val_preds_out = best_val_preds if best_val_preds else last_val_preds

    return model, (train_labels_out, train_preds_out), (val_labels_out, val_preds_out)


def parse_args():
    parser = argparse.ArgumentParser("Train ResNet classifier on CAR-DD")
    parser.add_argument("--train_images", default="../Dataset/train/", help="Path to training images root")
    parser.add_argument("--train_annotations", default="../Dataset/train.json", help="Path to training annotations")
    parser.add_argument("--val_images", default="../Dataset/val/", help="Path to validation images root")
    parser.add_argument("--val_annotations", default="../Dataset/val.json", help="Path to validation annotations")
    parser.add_argument(
        "--train_bundle",
        default="../DatasetPreprocessedResnet/train",
        help="Directory with preprocessed numpy bundle for training",
    )
    parser.add_argument(
        "--val_bundle",
        default="../DatasetPreprocessedResnet/val",
        help="Directory with preprocessed numpy bundle for validation",
    )
    parser.add_argument(
        "--use_raw",
        action="store_true",
        help="Load images/annotations directly instead of numpy bundles.",
    )
    parser.add_argument("--img_size", type=int, default=224, help="Target square size for raw loading")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument(
        "--rare_classes",
        type=int,
        nargs="*",
        default=[2, 4],
        help="0-based class indices considered rare for extra augmentation.",
    )
    parser.add_argument("--no_pretrained", action="store_true", help="Disable ImageNet pretraining")
    parser.add_argument("--freeze_backbone", action="store_true", help="Freeze all layers except the final FC head")
    parser.add_argument(
        "--model_name",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "wide_resnet50_2", "wide_resnet101_2"],
    )
    parser.add_argument("--output", default="car_dd_resnet.pth", help="Where to save the trained checkpoint")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_class_weights", action="store_true", help="Disable class weights in the loss")
    parser.add_argument(
        "--no_drop_unlabeled",
        action="store_true",
        help="Keep images that have no bounding-box labels (default is to drop them).",
    )
    parser.add_argument(
        "--rare_weight_scale",
        type=float,
        default=2.0,
        help="Multiplier applied to class weights for rare classes (loss balancing).",
    )
    parser.add_argument(
        "--rare_sample_boost",
        type=float,
        default=3.0,
        help="Multiplier applied to sampler weights for rare classes (oversampling).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_numpy_bundles = not args.use_raw
    pretrained = not args.no_pretrained
    drop_unlabeled = not args.no_drop_unlabeled

    # Pick normalization based on weights (ImageNet mean/std by default)
    weights, norm = get_resnet_weights(args.model_name, pretrained)
    mean, std = norm
    normalize = T.Normalize(mean=mean, std=std)

    # Augmentations: keep tensor-friendly transforms
    common_train_aug = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
            p=0.5,
        ),
        T.RandomRotation(degrees=15, fill=0),
        T.RandomPerspective(distortion_scale=0.25, p=0.4),
    ])

    rare_train_aug = T.Compose([
        T.RandomApply(
            [T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.08)],
            p=0.8,
        ),
        T.RandomRotation(degrees=25, fill=0),
    ])

    if use_numpy_bundles:
        base = CarDDNumpyDataset(bundle_dir=args.train_bundle, normalize=False, build_semantic=False)
        val_base = CarDDNumpyDataset(bundle_dir=args.val_bundle, normalize=False, build_semantic=False)
    else:
        base = CarDDDataset(
            images_dir=args.train_images,
            annotations_path=args.train_annotations,
            target_size=args.img_size,
            build_semantic=False,
            normalize=False,
        )
        val_base = CarDDDataset(
            images_dir=args.val_images,
            annotations_path=args.val_annotations,
            target_size=args.img_size,
            build_semantic=False,
            normalize=False,
        )

    cls_ds = CarDD_Cls(
        base_dataset=base,
        common_aug=common_train_aug,
        rare_aug=rare_train_aug,
        normalize=normalize,
        rare_classes=args.rare_classes,
        drop_unlabeled=drop_unlabeled,
    )

    val_ds = CarDD_Cls(
        base_dataset=val_base,
        common_aug=None,
        rare_aug=None,
        normalize=normalize,
        rare_classes=[],
        drop_unlabeled=drop_unlabeled,
    )

    print(f"Train samples (after dropping unlabeled): {len(cls_ds)}")
    print(f"Val samples   (after dropping unlabeled): {len(val_ds)}")

    if len(cls_ds) == 0:
        raise RuntimeError("No training samples available. Check paths or disable dropping unlabeled images.")
    if len(val_ds) == 0:
        raise RuntimeError("No validation samples available. Check paths or disable dropping unlabeled images.")

    labels = [sample[1].item() for sample in cls_ds]
    labels_np = np.array(labels, dtype=np.int64)
    class_counts = np.bincount(labels_np, minlength=args.num_classes)
    print("Class counts:", class_counts)

    class_weights_np = np.zeros(args.num_classes, dtype=np.float32)
    nonzero_mask = class_counts > 0
    if nonzero_mask.any():
        class_weights_np[nonzero_mask] = 1.0 / np.sqrt(class_counts[nonzero_mask])
        class_weights_np[nonzero_mask] = class_weights_np[nonzero_mask] / class_weights_np[nonzero_mask].mean()
    # Boost rare classes in the loss
    for rc in args.rare_classes:
        if 0 <= rc < args.num_classes:
            class_weights_np[rc] *= args.rare_weight_scale

    # Sampler weights (oversample rare classes)
    sample_weights = class_weights_np[labels_np]
    for idx, cls in enumerate(labels_np):
        if cls in args.rare_classes:
            sample_weights[idx] *= args.rare_sample_boost

    sample_weights_t = torch.from_numpy(sample_weights.astype(np.float32))
    class_weights_t = torch.from_numpy(class_weights_np.astype(np.float32)) if not args.no_class_weights else None

    sampler = WeightedRandomSampler(
        weights=sample_weights_t,
        num_samples=len(sample_weights_t),
        replacement=True,
    )

    train_loader = DataLoader(
        cls_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = build_resnet(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    model, (train_labels, train_preds), (val_labels, val_preds) = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=class_weights_t,
    )

    train_cm, train_prec, train_rec, train_f1 = per_class_prf1(train_labels, train_preds, num_classes=args.num_classes)
    val_cm, val_prec, val_rec, val_f1 = per_class_prf1(val_labels, val_preds, num_classes=args.num_classes)

    print("\nTrain Confusion Matrix:")
    print(train_cm)
    print_prf1(train_cm, train_prec, train_rec, train_f1, split="train")

    print("\nVal Confusion Matrix:")
    print(val_cm)
    print_prf1(val_cm, val_prec, val_rec, val_f1, split="val")

    # Highlight top confusions to inspect visually
    topc = top_confusions(val_cm, top_k=5)
    if topc:
        print("\nTop val confusions (count, true -> pred):")
        for count, t, p in topc:
            print(f"{count:4d} : {t} -> {p}")
    else:
        print("\nNo off-diagonal confusions found in val set.")

    torch.save(model.state_dict(), args.output)
    print(f"Saved checkpoint to {args.output}")


def per_class_prf1(labels: List[int], preds: List[int], num_classes: int):
    """Return per-class precision, recall, f1 as lists of length num_classes."""
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    tp = np.diag(cm).astype(np.float32)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp + 1e-9)
    recall = np.divide(tp, tp + fn + 1e-9)
    f1 = np.divide(2 * precision * recall, precision + recall + 1e-9)
    return cm, precision, recall, f1


def print_prf1(cm: np.ndarray, precision: np.ndarray, recall: np.ndarray, f1: np.ndarray, split: str):
    print(f"\nPer-class metrics ({split}):")
    print("class\tprec\trecall\tf1\t(tp/fp/fn)")
    for i in range(len(precision)):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        print(f"{i}\t{precision[i]:.3f}\t{recall[i]:.3f}\t{f1[i]:.3f}\t({tp}/{fp}/{fn})")


def top_confusions(cm: np.ndarray, top_k: int = 5):
    """Return list of (count, from_class, to_class) for the largest off-diagonal confusions."""
    conf_list = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                conf_list.append((cm[i, j], i, j))
    conf_list.sort(reverse=True, key=lambda x: x[0])
    return conf_list[:top_k]


if __name__ == "__main__":
    main()
