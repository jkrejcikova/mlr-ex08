import json
import os
import random
import sys
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import onnx

# ---------------------------------------------------------------------------
# Hardware & Configuration
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

color_geom_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class LeafDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.transform = transform
        self.classes = sorted(os.listdir(folder))
        self.samples = []
        for i, cls_name in enumerate(self.classes):
            cls_folder = os.path.join(folder, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for file_name in sorted(os.listdir(cls_folder)):
                file_path = os.path.join(cls_folder, file_name)
                if os.path.isfile(file_path):
                    self.samples.append((file_path, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CameraValDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def build_camera_val_dataset(val_folder="data/leaves/val", seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    cam_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    raw_ds = LeafDataset(val_folder, transform=None)
    samples = []
    for i in range(len(raw_ds)):
        img, label = raw_ds[i]
        samples.append((cam_transform(img), label))
    return CameraValDataset(samples)


def get_mobilenet():
    model = models.mobilenet_v3_small(weights="DEFAULT")
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


def train_model(model, train_loader, optimizer, criterion, n_epochs, scheduler):
    for epoch in range(n_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
    return model


# ---------------------------------------------------------------------------
# Main Training & Export Routine
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    val_ds = LeafDataset("data/leaves/val", transform=base_transform)
    test_old_ds = LeafDataset("data/leaves_old/val", transform=base_transform)
    cam_val_ds = build_camera_val_dataset("data/leaves/val", seed=42)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    cam_val_loader = DataLoader(cam_val_ds, batch_size=32, shuffle=False)
    test_old_loader = DataLoader(test_old_ds, batch_size=32, shuffle=False)

    train_ds = LeafDataset("data/leaves/train", transform=color_geom_transform)
    classes = train_ds.classes
    print(f"Classes in training dataset ({len(classes)}): {classes}")

    results = []
    models_dict = {}

    for seed in SEEDS:
        print(f"\nTraining Variant D with seed {seed}...")
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)

        model = get_mobilenet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

        train_model(model, train_loader, optimizer, criterion, n_epochs=30, scheduler=scheduler)

        # Save weights
        weight_path = f"model_D_seed{seed}.pt"
        torch.save(model.state_dict(), weight_path)
        print(f"Saved weights to {weight_path}")

        # Evaluate
        acc_a = evaluate(model, val_loader)
        acc_b = evaluate(model, cam_val_loader)
        acc_c = evaluate(model, test_old_loader)
        mean_abc = (acc_a + acc_b + acc_c) / 3.0

        res = {
            "seed": seed,
            "weight_path": weight_path,
            "acc_a": acc_a,
            "acc_b": acc_b,
            "acc_c": acc_c,
            "mean_abc": mean_abc
        }
        results.append(res)
        models_dict[seed] = model

        print(f"Seed {seed} Accuracies:")
        print(f"  (a) data/leaves/val:     {acc_a * 100:.1f}%")
        print(f"  (b) camera-like val:     {acc_b * 100:.1f}%")
        print(f"  (c) data/leaves_old/val: {acc_c * 100:.1f}%")
        print(f"  Mean (a, b, c):          {mean_abc * 100:.1f}%")

    # Select best model by mean(a, b, c)
    best_res = max(results, key=lambda x: x["mean_abc"])
    best_seed = best_res["seed"]
    best_model = models_dict[best_seed]
    print(f"\nBest model selected: Seed {best_seed} with mean(a, b, c) = {best_res['mean_abc'] * 100:.2f}%")

    # Export to ONNX exactly as cell 2e does
    onnx_file_name = "leaf_model.onnx"
    print(f"\nExporting best model (seed {best_seed}) to {onnx_file_name}...")
    best_model.eval().cpu()
    torch.onnx.export(
        best_model,
        (torch.zeros(1, 3, 96, 96),),
        onnx_file_name,
        input_names=["image"],
        output_names=["logits"],
        external_data=False
    )

    model_info = {"classes": classes, "input_size": 96, "mean": mean, "std": std}
    onnx_model = onnx.load(onnx_file_name)
    onnx_model.metadata_props.add(key="model_info", value=json.dumps(model_info))
    onnx.save(onnx_model, onnx_file_name)

    onnx_size_mb = os.path.getsize(onnx_file_name) / 1e6
    full_path = os.path.abspath(onnx_file_name)
    print(f"ONNX export successful!")
    print(f"  File name:  {onnx_file_name}")
    print(f"  Path:       {full_path}")
    print(f"  Size:       {onnx_size_mb:.2f} MB")
    print(f"  Class order in metadata: {classes}")
