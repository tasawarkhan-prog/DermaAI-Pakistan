# train_colab.py
# ============================================================
# DermaAI Pakistan — EfficientNet-B0 Fine-Tuning on HAM10000
# Run this in Google Colab (free T4 GPU) or Kaggle (~15-20 min)
# After training, upload model to HuggingFace Hub for inference
# ============================================================
# USAGE:
#   1. Open Google Colab: colab.research.google.com
#   2. Runtime → Change Runtime → T4 GPU
#   3. Run all cells (Runtime → Run All)
#   4. Model saves to /content/dermaai_efficientnet_b0.pth
#   5. Upload to HuggingFace Hub (instructions at end)
# ============================================================

# ─── CELL 1: Install dependencies ───────────────────────────
"""
!pip install -q torch torchvision timm kaggle --upgrade
!pip install -q pytorch-grad-cam scikit-learn matplotlib seaborn
"""

# ─── CELL 2: Download HAM10000 from Kaggle ──────────────────
"""
# Option A — Kaggle API (recommended)
# 1. Create kaggle.json from kaggle.com/account
# 2. Upload it to Colab Files

import os
from google.colab import files

uploaded = files.upload()  # upload kaggle.json
os.makedirs('/root/.config/kaggle', exist_ok=True)
os.system('cp kaggle.json /root/.config/kaggle/')
os.system('chmod 600 /root/.config/kaggle/kaggle.json')
os.system('kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection -p /content --unzip')

# Option B — ISIC Archive (manual download links)
# https://isic-archive.com/api/v1/download/
"""

# ─── CELL 3: Imports ─────────────────────────────────────────
import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

# ─── CELL 4: Configuration ───────────────────────────────────
CONFIG = {
    "data_dir": "/content/HAM10000_images",   # adjust to actual path
    "metadata_csv": "/content/HAM10000_metadata.csv",
    "output_dir": "/content/dermaai_checkpoints",
    "model_name": "efficientnet_b0",
    "num_classes": 7,
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 2,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_every": 5,
}

# HAM10000 class mapping
CLASS_NAMES = {
    "akiec": 0,  # Actinic Keratoses
    "bcc":   1,  # Basal Cell Carcinoma
    "bkl":   2,  # Benign Keratosis
    "df":    3,  # Dermatofibroma
    "mel":   4,  # Melanoma
    "nv":    5,  # Melanocytic Nevi
    "vasc":  6,  # Vascular Lesions
}

CLASS_LABELS = {v: k for k, v in CLASS_NAMES.items()}
CLASS_FULL_NAMES = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}

# Map HAM10000 classes to DermaAI condition keys
HAM_TO_DERMAAI = {
    "akiec": "contact_dermatitis",   # Actinic → related
    "bcc":   "melanoma",             # BCC → skin cancer group
    "bkl":   "rosacea",              # Benign → closest match
    "df":    "contact_dermatitis",
    "mel":   "melanoma",
    "nv":    "vitiligo",             # Nevi → pigmentation
    "vasc":  "urticaria",            # Vascular → closest
}

print(f"Device: {CONFIG['device']}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(CONFIG["output_dir"], exist_ok=True)
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# ─── CELL 5: Dataset ─────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["image_id"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = CLASS_NAMES[row["dx"]]
        return image, label


def build_transforms(image_size: int, augment: bool = False):
    if augment:
        return T.Compose([
            T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomRotation(30),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return T.Compose([
            T.Resize(int(image_size * 1.1)),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def prepare_data(config):
    df = pd.read_csv(config["metadata_csv"])
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:\n{df['dx'].value_counts()}")

    # Deduplicate by lesion_id (keep one image per lesion)
    df = df.groupby("lesion_id").first().reset_index()
    print(f"After dedup: {len(df)} unique lesions")

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["dx"], random_state=config["seed"]
    )

    train_ds = HAM10000Dataset(train_df, config["data_dir"], build_transforms(config["image_size"], augment=True))
    val_ds   = HAM10000Dataset(val_df,   config["data_dir"], build_transforms(config["image_size"], augment=False))

    # Class weights for imbalanced dataset
    class_counts = df["dx"].map(CLASS_NAMES).value_counts().sort_index()
    class_weights = 1.0 / class_counts.values.astype(float)
    class_weights = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=True)

    return train_loader, val_loader, class_weights


# ─── CELL 6: Model ───────────────────────────────────────────
def build_model(config):
    model = timm.create_model(
        config["model_name"],
        pretrained=True,
        num_classes=config["num_classes"],
    )
    # Freeze early layers for faster training
    for name, param in model.named_parameters():
        if "blocks.0" in name or "blocks.1" in name:
            param.requires_grad = False
    model = model.to(config["device"])
    print(f"Model: {config['model_name']} | Params: {sum(p.numel() for p in model.parameters()):,}")
    return model


# ─── CELL 7: Training loop ───────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(config):
    print("=" * 60)
    print("DermaAI Pakistan — EfficientNet-B0 Training")
    print("=" * 60)

    train_loader, val_loader, class_weights = prepare_data(config)
    model = build_model(config)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["device"]))
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config["device"])
        val_loss, val_acc, preds, labels = val_epoch(model, val_loader, criterion, config["device"])
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": config,
                "class_names": CLASS_NAMES,
                "class_labels": CLASS_LABELS,
                "class_full_names": CLASS_FULL_NAMES,
                "ham_to_dermaai": HAM_TO_DERMAAI,
            }, os.path.join(config["output_dir"], "best_model.pth"))
            print(f"  ★ New best model saved! Val Acc: {val_acc:.4f}")

        if epoch % config["save_every"] == 0:
            torch.save(model.state_dict(),
                       os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch}.pth"))

    # Final classification report
    print("\n" + "=" * 60)
    print("Final Classification Report (Validation Set):")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=list(CLASS_FULL_NAMES.values())))

    # Save history
    with open(os.path.join(config["output_dir"], "history.json"), "w") as f:
        json.dump(history, f)

    # Plot
    _plot_training(history, config["output_dir"])
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {config['output_dir']}/best_model.pth")
    return model


def _plot_training(history, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss Curves"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_title("Accuracy Curves"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Training curves saved.")


# ─── CELL 8: Upload to HuggingFace Hub ──────────────────────
def upload_to_hf_hub(model_path: str, repo_id: str, hf_token: str):
    """
    Upload trained model to HuggingFace Hub for use in the Streamlit app.

    Args:
        model_path: Path to best_model.pth
        repo_id: e.g. 'your-username/dermaai-pakistan-model'
        hf_token: HuggingFace token from huggingface.co/settings/tokens
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True, private=False)
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="best_model.pth",
            repo_id=repo_id,
            token=hf_token,
        )
        print(f"Model uploaded to: https://huggingface.co/{repo_id}")
        print(f"Update MODEL_HF_REPO in app.py to: '{repo_id}'")
    except ImportError:
        print("Install: pip install huggingface_hub")


# ─── CELL 9: Run training ────────────────────────────────────
if __name__ == "__main__":
    trained_model = train(CONFIG)

    # Uncomment and fill in to upload to HF Hub:
    # HF_TOKEN = "hf_..."   # from huggingface.co/settings/tokens
    # HF_REPO  = "your-username/dermaai-pakistan"
    # upload_to_hf_hub(
    #     model_path=os.path.join(CONFIG["output_dir"], "best_model.pth"),
    #     repo_id=HF_REPO,
    #     hf_token=HF_TOKEN,
    # )
