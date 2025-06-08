import os
import json
import torch
import logging
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.optim import AdamW
from transformers import ViTConfig, ViTForImageClassification

from src.config import Config
from src.codes.data import get_train_loader, get_val_loader

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_model():
    config = ViTConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        image_size=Config.IMAGE_SIZE,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        num_labels=50
    )
    model = ViTForImageClassification(config)

    with torch.no_grad():
        model.classifier.weight[40:].requires_grad = False
        model.classifier.bias[40:].requires_grad = False

    return model


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            outputs[:, 40:] = -1e9  # Mask future classes
            loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def load_progress(progress_path):
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            progress = json.load(f)
        return progress.get("best_val_acc", 0.0), progress.get("last_epoch", 0)
    return 0.0, 0


def save_progress(progress_path, best_val_acc, epoch):
    progress = {
        "best_val_acc": best_val_acc,
        "last_epoch": epoch
    }
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=4)


def train():
    device = Config.DEVICE
    outdir = Path(Config.TRAIN.OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("🧠 Starting ViT training...")
    logger.info(f"📁 Output directory: {outdir}")
    logger.info(f"🖥️ Device: {device}")

    train_loader = get_train_loader()
    val_loader = get_val_loader()

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.TRAIN.LR,
        weight_decay=Config.TRAIN.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.TRAIN.EPOCHS)

    best_model_path = Config.TRAIN.model_path()
    progress_path = Config.TRAIN.progress_path()

    best_val_acc, last_epoch = 0.0, 0
    start_epoch = 1

    # Resume logic
    if getattr(Config.TRAIN, "RESUME", False) and os.path.exists(best_model_path):
        logger.info(f"🔁 Resuming from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)

        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            logger.info("✅ Loaded full training state.")

            if os.path.exists(progress_path):
                best_val_acc, last_epoch = load_progress(progress_path)
                logger.info(f"📄 Loaded progress.json — Epoch {last_epoch}, Val Acc: {best_val_acc:.2f}%")
            else:
                last_epoch = checkpoint.get("epoch", 0)
                val_acc, _ = evaluate(model, val_loader, device)
                best_val_acc = val_acc
                save_progress(progress_path, best_val_acc, last_epoch)
                logger.info(f"📄 Created progress.json — Epoch {last_epoch}, Val Acc: {best_val_acc:.2f}%")

            start_epoch = last_epoch + 1

        else:
            logger.info("⚠️ Checkpoint is model-only. Loading weights and evaluating...")
            model.load_state_dict(checkpoint)
            val_acc, _ = evaluate(model, val_loader, device)
            best_val_acc = val_acc
            last_epoch = 0
            save_progress(progress_path, best_val_acc, last_epoch)
            logger.info(f"📄 Initialized progress.json at epoch {last_epoch} with Val Acc: {best_val_acc:.2f}%")
            start_epoch = last_epoch + 1

    for epoch in range(start_epoch, Config.TRAIN.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.TRAIN.EPOCHS}", unit="batch")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).logits
            outputs[:, 40:] = -1e9
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        logger.info(f"📉 Epoch {epoch} — Training Loss: {avg_loss:.4f}")

        val_acc, val_loss = evaluate(model, val_loader, device)
        logger.info(f"🧪 Epoch {epoch} — Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_val_acc': best_val_acc
            }, best_model_path)
            save_progress(progress_path, best_val_acc, epoch)
            logger.info(f"💾 New best model saved! Val Acc: {val_acc:.2f}% → {best_model_path}")

    logger.info("✅ Training complete!")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"📁 Final model saved at: {best_model_path}")


if __name__ == "__main__":
    train()
