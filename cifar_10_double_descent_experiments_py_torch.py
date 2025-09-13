#!/usr/bin/env python
"""
CIFAR-10 Double Descent: model-wise and epoch-wise experiments (PyTorch)

Usage examples:

# 1) Model-wise double descent sweep over width multipliers
python cifar10_double_descent.py \
  --experiment modelwise \
  --widths 1 2 4 8 16 \
  --epochs 200 \
  --batch-size 128 \
  --lr 0.1 \
  --label-noise 0.2 \
  --train-fraction 0.2 \
  --repeats 3 \
  --results-dir results_modelwise

# 2) Epoch-wise double descent on a fixed (overparameterized) model
python cifar10_double_descent.py \
  --experiment epochwise \
  --width 8 \
  --epochs 600 \
  --batch-size 128 \
  --lr 0.1 \
  --label-noise 0.2 \
  --train-fraction 0.2 \
  --results-dir results_epochwise

Notes
-----
- To make double descent more visible and reproducible, we deliberately:
  * minimize explicit regularization (no weight decay, no dropout)
  * keep augmentations minimal (normalize only)
  * optionally add label noise and/or reduce training set size (train_fraction)
  * train long enough so some models reach (near) 100% training accuracy
- You may need to tune widths/epochs/noise depending on your hardware to clearly
  see the rise near the interpolation threshold.
"""
import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback if tqdm isn't available

# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------
# Data with optional label noise and subsampling
# ------------------------------

def load_cifar10(data_dir: str, train_fraction: float = 1.0, label_noise: float = 0.0, seed: int = 42):
    """Load CIFAR-10 with minimal augmentation and optional label noise/subsampling."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # Subsample training set
    if train_fraction < 1.0:
        set_seed(seed)
        n = len(train)
        keep = int(n * train_fraction)
        idx = torch.randperm(n)[:keep].tolist()
        train = Subset(train, idx)

    # Apply label noise by randomly reassigning a fraction of labels
    if label_noise > 0.0:
        if isinstance(train, Subset):
            targets = [train.dataset.targets[i] for i in train.indices]
        else:
            targets = list(train.targets)

        set_seed(seed + 999)
        num_noisy = int(len(targets) * label_noise)
        noisy_idx = torch.randperm(len(targets))[:num_noisy].tolist()
        for i in noisy_idx:
            targets[i] = random.randint(0, 9)

        # write back
        if isinstance(train, Subset):
            for j, i in enumerate(train.indices):
                train.dataset.targets[i] = targets[j]
        else:
            train.targets = targets

    return train, test


# ------------------------------
# Simple ConvNet with width multiplier
# ------------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:
            x = self.shortcut(x)
        out = F.relu(out + x)
        return out


class SimpleCNN(nn.Module):
    """A small ResNet-ish CNN where capacity scales primarily with width."""
    def __init__(self, width: int = 1, num_classes: int = 10, base_channels: int = 16):
        super().__init__()
        c1 = base_channels * width
        c2 = base_channels * 2 * width
        c3 = base_channels * 4 * width

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.layer1 = nn.Sequential(
            BasicBlock(c1, c1),
            BasicBlock(c1, c1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(c1, c2, stride=2),
            BasicBlock(c2, c2),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(c2, c3, stride=2),
            BasicBlock(c3, c3),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c3, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


# ------------------------------
# Train / Eval
# ------------------------------

def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(images)
                loss = F.cross_entropy(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += loss.item() * images.size(0)
        pred = outputs.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            running_loss += loss.item() * images.size(0)
            pred = outputs.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    return running_loss / total, correct / total


# ------------------------------
# Experiments
# ------------------------------

@dataclass
class Config:
    experiment: str
    widths: List[int]
    width: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    label_noise: float
    train_fraction: float
    data_dir: str
    results_dir: str
    repeats: int
    seed: int
    use_amp: bool


def run_modelwise(cfg: Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    train_ds, test_ds = load_cifar10(cfg.data_dir, cfg.train_fraction, cfg.label_noise, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    lines = ["width,params,repeat,epoch,train_loss,train_acc,test_loss,test_acc"]

    for w in cfg.widths:
        for r in range(cfg.repeats):
            set_seed(cfg.seed + 1000 * r + w)
            model = SimpleCNN(width=w).to(device)
            params = count_parameters(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
            scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

            pbar = tqdm(range(1, cfg.epochs + 1), desc=f"width={w} rep={r}")
            for epoch in pbar:
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
                test_loss, test_acc = evaluate(model, test_loader, device)
                scheduler.step()
                pbar.set_postfix({"train_acc": f"{train_acc*100:.1f}", "test_acc": f"{test_acc*100:.1f}"})
                lines.append(f"{w},{params},{r},{epoch},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},{test_acc:.6f}")

    csv_path = os.path.join(cfg.results_dir, "modelwise_log.csv")
    with open(csv_path, 'w') as f:
        f.write("\n".join(lines))

    # Plot final-epoch curve (test error vs params)
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(csv_path)
        # take last epoch per (width, repeat)
        last = df.sort_values('epoch').groupby(['width', 'repeat']).tail(1)
        agg = last.groupby('width').agg({'test_acc': 'mean', 'params': 'mean'}).reset_index()
        agg['test_error'] = 1 - agg['test_acc']
        plt.figure()
        plt.plot(agg['params'], agg['test_error'], marker='o')
        plt.xscale('log')
        plt.xlabel('Number of parameters (log scale)')
        plt.ylabel('Test error')
        plt.title('Model-wise double descent (final epoch)')
        plt.grid(True, which='both', ls='--', alpha=0.3)
        fig1 = os.path.join(cfg.results_dir, 'modelwise_test_error_vs_params.png')
        plt.savefig(fig1, dpi=160, bbox_inches='tight')

        # Identify (approximate) interpolation threshold: first width where train_acc ~ 1.0
        interp = last.groupby('width')['train_acc'].mean()
        interp_width = None
        for w in agg['width']:
            if interp.get(w, 0) >= 0.999:
                interp_width = w
                break
        if interp_width is not None:
            print(f"[INFO] Approx interpolation threshold near width={interp_width}.")
        else:
            print("[WARN] No width reached ~100% train accuracy; consider larger widths/epochs.")

    except Exception as e:
        print(f"Plotting skipped due to: {e}")


def run_epochwise(cfg: Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, test_ds = load_cifar10(cfg.data_dir, cfg.train_fraction, cfg.label_noise, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    set_seed(cfg.seed)
    model = SimpleCNN(width=cfg.width).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.use_amp and device.type == 'cuda')

    lines = ["width,params,epoch,train_loss,train_acc,test_loss,test_acc"]

    pbar = tqdm(range(1, cfg.epochs + 1), desc=f"epochwise width={cfg.width}")
    for epoch in pbar:
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device, scaler)
        test_loss, test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        pbar.set_postfix({"train_acc": f"{train_acc*100:.1f}", "test_acc": f"{test_acc*100:.1f}"})
        lines.append(f"{cfg.width},{params},{epoch},{train_loss:.6f},{train_acc:.6f},{test_loss:.6f},{test_acc:.6f}")

    csv_path = os.path.join(cfg.results_dir, "epochwise_log.csv")
    with open(csv_path, 'w') as f:
        f.write("\n".join(lines))

    # Plot test error vs epoch
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.read_csv(csv_path)
        df['test_error'] = 1 - df['test_acc']
        plt.figure()
        plt.plot(df['epoch'], df['test_error'], marker='.', linewidth=1)
        plt.xlabel('Epoch')
        plt.ylabel('Test error')
        plt.title('Epoch-wise double descent (fixed model)')
        plt.grid(True, ls='--', alpha=0.3)
        fig = os.path.join(cfg.results_dir, 'epochwise_test_error_vs_epoch.png')
        plt.savefig(fig, dpi=160, bbox_inches='tight')
    except Exception as e:
        print(f"Plotting skipped due to: {e}")


# ------------------------------
# Main
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CIFAR-10 Double Descent Experiments")
    p.add_argument('--experiment', type=str, choices=['modelwise', 'epochwise'], required=True)

    # Model-wise sweep
    p.add_argument('--widths', type=int, nargs='*', default=[1, 2, 4, 8, 16], help='Width multipliers to sweep')
    p.add_argument('--repeats', type=int, default=1, help='How many random-seed repeats per width (>=3 recommended)')

    # Epoch-wise
    p.add_argument('--width', type=int, default=8, help='Width for epoch-wise experiment')

    # Common training hyperparams
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=0.0)

    # Data knobs to expose interpolation threshold
    p.add_argument('--label-noise', type=float, default=0.0, help='Fraction of training labels to randomize [0,1)')
    p.add_argument('--train-fraction', type=float, default=1.0, help='Use only this fraction of the training set (0 < f <= 1)')

    # Misc
    p.add_argument('--data-dir', type=str, default='./data')
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-amp', action='store_true', help='Disable CUDA AMP (mixed precision)')

    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        experiment=args.experiment,
        widths=args.widths,
        width=args.width,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_noise=args.label_noise,
        train_fraction=args.train_fraction,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        repeats=args.repeats,
        seed=args.seed,
        use_amp=not args.no_amp,
    )

    print(cfg)

    if cfg.experiment == 'modelwise':
        run_modelwise(cfg)
    elif cfg.experiment == 'epochwise':
        run_epochwise(cfg)
    else:
        raise ValueError(cfg.experiment)


if __name__ == '__main__':
    main()
