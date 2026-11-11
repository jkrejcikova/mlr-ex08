import copy
import csv
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Hardware & Configuration
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Base evaluation transform (identical to Task 2 & 3)
base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Geometric augmentation (Task 3d)
geom_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Geometric + ColorJitter augmentation
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
    """Dataset from Task 3a: loads sorted class folders and image files."""
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
    """Holds pre-generated camera-like validation images."""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def build_camera_val_dataset(val_folder="data/leaves/val", seed=42):
    """Generates the camera-like validation set once with a fixed seed."""
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


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class LeafNet(nn.Module):
    """Baseline architecture from Task 3b."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeafNetBN(nn.Module):
    """Variant 3: LeafNet with BatchNorm2d after convolutions (Task 3d)."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))

        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LeafNetDropout(nn.Module):
    """Variant 4: LeafNet with Dropout (p=0.2) after flatten and after fc1."""
    def __init__(self, p=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(p)

        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        x = x.flatten(1)
        x = self.drop(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x


class LeafNetGAP(nn.Module):
    """Variant 5: Global Average Pooling instead of flatten."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))

        x = self.gap(x).flatten(1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_mobilenet(freeze_base=True):
    """Variants 7 & 8: Pretrained MobileNetV3-Small from Task 2b."""
    model = models.mobilenet_v3_small(weights="DEFAULT")
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    return model


# ---------------------------------------------------------------------------
# Evaluation & Training loop
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    """Computes top-1 accuracy on a given loader."""
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


def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs,
                scheduler=None, keep_best_weights=False):
    """Standard Task 3c training loop with optional scheduler and best-weight checkpointing."""
    best_acc = -1.0
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        loss_sum = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if scheduler is not None:
            scheduler.step()

        if keep_best_weights:
            val_acc = evaluate(model, val_loader)
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = copy.deepcopy(model.state_dict())

    if keep_best_weights and best_weights is not None:
        model.load_state_dict(best_weights)

    return model


# ---------------------------------------------------------------------------
# Runner for a single variant over SEEDS
# ---------------------------------------------------------------------------
def run_variant(variant_name, cam_val_ds, seeds=SEEDS, best_variant_config=None):
    """Executes a variant for all seeds, evaluates train, (a), (b), (c), and returns stats."""
    val_ds = LeafDataset("data/leaves/val", transform=base_transform)
    test_old_ds = LeafDataset("data/leaves_old/val", transform=base_transform)

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    cam_val_loader = DataLoader(cam_val_ds, batch_size=32, shuffle=False)
    test_old_loader = DataLoader(test_old_ds, batch_size=32, shuffle=False)

    acc_train_list = []
    acc_a_list = []
    acc_b_list = []
    acc_c_list = []

    start_time = time.time()

    for seed in seeds:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        g = torch.Generator()
        g.manual_seed(seed)

        train_folder = "data/leaves/train"
        use_combined_data = False
        n_epochs = 15
        scheduler_fn = None
        keep_best = False
        criterion = nn.CrossEntropyLoss()

        if variant_name == "baseline":
            train_transform = base_transform
            model = LeafNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "1_geometric_aug":
            train_transform = geom_transform
            model = LeafNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "2_geometric_color_aug":
            train_transform = color_geom_transform
            model = LeafNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "3_batch_norm":
            train_transform = base_transform
            model = LeafNetBN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "4_dropout":
            train_transform = base_transform
            model = LeafNetDropout(p=0.2)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "5_gap":
            train_transform = base_transform
            model = LeafNetGAP()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "6_adamw":
            train_transform = base_transform
            model = LeafNet()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

        elif variant_name == "7_mobilenet_frozen":
            train_transform = base_transform
            model = get_mobilenet(freeze_base=True)
            optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=1e-3)

        elif variant_name == "8_mobilenet_finetuned":
            train_transform = base_transform
            model = get_mobilenet(freeze_base=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "9a_mobilenet_aug":
            train_transform = geom_transform
            model = get_mobilenet(freeze_base=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        elif variant_name == "9b_mobilenet_aug_cosine_30ep":
            train_transform = geom_transform
            n_epochs = 30
            model = get_mobilenet(freeze_base=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        elif variant_name == "10_combined_data":
            use_combined_data = True
            # Use best variant setup if provided, else MobileNet finetuned + aug + Cosine
            if best_variant_config == "8_mobilenet_finetuned":
                train_transform = base_transform
                model = get_mobilenet(freeze_base=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            elif best_variant_config == "9a_mobilenet_aug":
                train_transform = geom_transform
                model = get_mobilenet(freeze_base=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            elif best_variant_config == "9b_mobilenet_aug_cosine_30ep":
                train_transform = geom_transform
                n_epochs = 30
                model = get_mobilenet(freeze_base=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)
            else:
                # Default to best architecture
                train_transform = geom_transform
                n_epochs = 30
                model = get_mobilenet(freeze_base=False)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        elif variant_name == "11_label_smoothing":
            train_transform = base_transform
            model = LeafNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        elif variant_name == "12_best_epoch_weights":
            train_transform = base_transform
            model = LeafNet()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            keep_best = True

        elif variant_name == "comb_A":
            # Combination A: MobileNet (frozen rest) + geom + ColorJitter + CosineAnnealingLR, 30 epochs
            train_transform = color_geom_transform
            n_epochs = 30
            model = get_mobilenet(freeze_base=True)
            optimizer = torch.optim.Adam(model.classifier[3].parameters(), lr=1e-3)
            scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        elif variant_name == "comb_B":
            # Combination B: MobileNet fine-tuned + geom aug (same as 9b), 30 epochs, constant lr (NO scheduler)
            train_transform = geom_transform
            n_epochs = 30
            model = get_mobilenet(freeze_base=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler_fn = None

        elif variant_name == "comb_C":
            # Combination C: LeafNetBN from scratch + geom + ColorJitter + CosineAnnealingLR, 30 epochs
            train_transform = color_geom_transform
            n_epochs = 30
            model = LeafNetBN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        elif variant_name == "comb_D":
            # Combination D: MobileNet fine-tuned + geom + ColorJitter + CosineAnnealingLR, 30 epochs
            train_transform = color_geom_transform
            n_epochs = 30
            model = get_mobilenet(freeze_base=False)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler_fn = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=30)

        else:
            raise ValueError(f"Unknown variant {variant_name}")

        model = model.to(device)
        scheduler = scheduler_fn(optimizer) if scheduler_fn else None

        # Build train dataset and training loader
        if use_combined_data:
            ds_cur = LeafDataset("data/leaves/train", transform=train_transform)
            ds_old = LeafDataset("data/leaves_old/train", transform=train_transform)
            train_ds = ConcatDataset([ds_cur, ds_old])

            ds_cur_eval = LeafDataset("data/leaves/train", transform=base_transform)
            ds_old_eval = LeafDataset("data/leaves_old/train", transform=base_transform)
            train_eval_ds = ConcatDataset([ds_cur_eval, ds_old_eval])
        else:
            train_ds = LeafDataset(train_folder, transform=train_transform)
            train_eval_ds = LeafDataset(train_folder, transform=base_transform)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=g)
        train_eval_loader = DataLoader(train_eval_ds, batch_size=32, shuffle=False)

        # Train
        train_model(model, train_loader, val_loader, optimizer, criterion,
                    n_epochs=n_epochs, scheduler=scheduler, keep_best_weights=keep_best)

        # Evaluate at the end
        acc_train = evaluate(model, train_eval_loader)
        acc_a = evaluate(model, val_loader)
        acc_b = evaluate(model, cam_val_loader)
        acc_c = evaluate(model, test_old_loader)

        acc_train_list.append(acc_train)
        acc_a_list.append(acc_a)
        acc_b_list.append(acc_b)
        acc_c_list.append(acc_c)

    elapsed = time.time() - start_time

    def format_stats(arr):
        mean_val = np.mean(arr) * 100
        min_val = np.min(arr) * 100
        max_val = np.max(arr) * 100
        return f"{mean_val:5.1f}% [{min_val:4.1f}% - {max_val:4.1f}%]"

    def format_seeds(arr):
        return ", ".join(f"{x * 100:4.1f}%" for x in arr)

    result = {
        "variant": variant_name,
        "train_acc": format_stats(acc_train_list),
        "leaves_val (a)": format_stats(acc_a_list),
        "camera_val (b)": format_stats(acc_b_list),
        "leaves_old_val (c)": format_stats(acc_c_list),
        "train_seeds": format_seeds(acc_train_list),
        "a_seeds": format_seeds(acc_a_list),
        "b_seeds": format_seeds(acc_b_list),
        "c_seeds": format_seeds(acc_c_list),
        "elapsed_sec": round(elapsed, 2),
        "raw_train": acc_train_list,
        "raw_a": acc_a_list,
        "raw_b": acc_b_list,
        "raw_c": acc_c_list,
    }

    if variant_name == "10_combined_data":
        result["leaves_old_val (c)"] += " (* not fully unseen)"

    return result


def append_to_csv(result, csv_path="results.csv"):
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Variant",
                "Train Acc (mean [min-max])",
                "leaves/val (a) (mean [min-max])",
                "camera-like (b) (mean [min-max])",
                "leaves_old/val (c) (mean [min-max])",
                "Train Acc (seeds)",
                "leaves/val (a) (seeds)",
                "camera-like (b) (seeds)",
                "leaves_old/val (c) (seeds)",
                "Time (s)"
            ])
        writer.writerow([
            result["variant"],
            result["train_acc"],
            result["leaves_val (a)"],
            result["camera_val (b)"],
            result["leaves_old_val (c)"],
            result["train_seeds"],
            result["a_seeds"],
            result["b_seeds"],
            result["c_seeds"],
            result["elapsed_sec"]
        ])


# ---------------------------------------------------------------------------
# Main Execution: run all variants in order
# ---------------------------------------------------------------------------
ALL_VARIANTS_ORDER = [
    "baseline",
    "1_geometric_aug",
    "2_geometric_color_aug",
    "3_batch_norm",
    "4_dropout",
    "5_gap",
    "6_adamw",
    "7_mobilenet_frozen",
    "8_mobilenet_finetuned",
    "9a_mobilenet_aug",
    "9b_mobilenet_aug_cosine_30ep",
    "11_label_smoothing",
    "12_best_epoch_weights",
    "10_combined_data"
]

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("Generating camera-like validation set (once with seed=42)...")
    cam_val_ds = build_camera_val_dataset("data/leaves/val", seed=42)
    print(f"Camera-like validation set ready: {len(cam_val_ds)} images.")

    if arg == "all":
        # Reset results.csv
        if os.path.exists("results.csv"):
            os.remove("results.csv")

        results_list = []
        best_name = None
        best_cam_score = -1.0

        for var in ALL_VARIANTS_ORDER:
            print(f"\n>>> Running: {var} ...")
            if var == "10_combined_data":
                # Feed the best variant configuration determined so far
                print(f"    (Variant 10 using best architecture: {best_name})")
                res = run_variant(var, cam_val_ds, seeds=SEEDS, best_variant_config=best_name)
            else:
                res = run_variant(var, cam_val_ds, seeds=SEEDS)
                # Keep track of best variant by camera-like val (or val)
                mean_cam = np.mean(res["raw_b"])
                if mean_cam > best_cam_score:
                    best_cam_score = mean_cam
                    best_name = var

            append_to_csv(res)
            results_list.append(res)
            print(f"    Train:       {res['train_acc']}")
            print(f"    (a) Val:     {res['leaves_val (a)']}")
            print(f"    (b) Camera:  {res['camera_val (b)']}")
            print(f"    (c) Old Val: {res['leaves_old_val (c)']}")
            print(f"    Time:        {res['elapsed_sec']:.2f} s")

        print("\n" + "=" * 95)
        print("EXPERIMENTS COMPLETE - ALL VARIANTS RECORDED IN results.csv")
        print("=" * 95)

    elif arg == "combinations":
        combo_list = ["comb_A", "comb_B", "comb_C"]
        for var in combo_list:
            print(f"\n>>> Running: {var} ...")
            res = run_variant(var, cam_val_ds, seeds=SEEDS)
            append_to_csv(res)
            print(f"    Train:       {res['train_acc']}")
            print(f"    (a) Val:     {res['leaves_val (a)']}")
            print(f"    (b) Camera:  {res['camera_val (b)']}")
            print(f"    (c) Old Val: {res['leaves_old_val (c)']}")
            print(f"    Time:        {res['elapsed_sec']:.2f} s")
        print("\n" + "=" * 95)
        print("COMBINATIONS COMPLETE - APPENDED TO results.csv")
        print("=" * 95)

    else:
        res = run_variant(arg, cam_val_ds, seeds=SEEDS)
        append_to_csv(res)
        print(f"\nResults for {arg}:")
        print(f"  Train:       {res['train_acc']} | seeds: {res['train_seeds']}")
        print(f"  (a) Val:     {res['leaves_val (a)']} | seeds: {res['a_seeds']}")
        print(f"  (b) Camera:  {res['camera_val (b)']} | seeds: {res['b_seeds']}")
        print(f"  (c) Old Val: {res['leaves_old_val (c)']} | seeds: {res['c_seeds']}")
        print(f"  Time:        {res['elapsed_sec']:.2f} s")
