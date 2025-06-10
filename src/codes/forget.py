import os
import json
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from itertools import cycle
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.codes.data import get_dynamic_loader
from src.pkgs.gs.vit_pytorch_face.vit_face import ViTClassifier

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USE_LORA = True


def get_model():
    return ViTClassifier(
        num_classes=50,  # Always 50 outputs
        image_size=Config.IMAGE_SIZE,
        patch_size=16,
        ac_patch_size=16,
        pad=0,
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        use_lora=USE_LORA,
        lora_rank=8 if USE_LORA else None
    )


def load_base_model_weights(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at {path}")
    logger.info(f"🔍 Loading base model from {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    logger.info("✅ Weights loaded with strict=False")
    return model


def enable_lora_training(model):
    total_params = sum(p.numel() for p in model.parameters())
    lora_params = 0

    for p in model.parameters():
        p.requires_grad = False

    for name, module in model.named_modules():
        for attr in ['lora_A', 'lora_B', 'lora_right', 'lora_down', 'lora_up']:
            if hasattr(module, attr):
                param_or_tensor = getattr(module, attr)
                if isinstance(param_or_tensor, nn.Parameter):
                    param_or_tensor.requires_grad = True
                    lora_params += param_or_tensor.numel()
                elif isinstance(param_or_tensor, nn.Module):
                    for p in param_or_tensor.parameters():
                        p.requires_grad = True
                        lora_params += p.numel()

    if lora_params == 0:
        print("⚠️ No LoRA parameters found!")
    else:
        percent = 100 * lora_params / total_params
        print(f"🔓 Training LoRA only: {lora_params:,} / {total_params:,} ({percent:.2f}%)")


def get_dataloaders():
    train_path = Config.FORGET.TRAIN_DATA_PATH
    val_path = Config.FORGET.VAL_DATA_PATH

    forget_classes = list(range(0, 10))
    retain_classes = list(range(10, 40))

    train_forget_loader = get_dynamic_loader(
        data_path=train_path,
        class_range=forget_classes,
        mode='train',
        use_original_labels=True
    )
    train_retain_loader = get_dynamic_loader(
        data_path=train_path,
        class_range=retain_classes,
        mode='train',
        use_original_labels=True
    )
    val_forget_loader = get_dynamic_loader(
        data_path=val_path,
        class_range=forget_classes,
        mode='val',
        use_original_labels=True
    )
    val_retain_loader = get_dynamic_loader(
        data_path=val_path,
        class_range=retain_classes,
        mode='val',
        use_original_labels=True
    )

    return train_forget_loader, train_retain_loader, val_forget_loader, val_retain_loader


def compute_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def compute_hmean(acc_retain, acc_forget):
    retain_ratio = acc_retain / 100.0
    forget_complement = 1 - acc_forget / 100.0
    denom = retain_ratio + forget_complement
    return 2 * retain_ratio * forget_complement / denom * 100.0 if denom != 0 else 0.0


def retention_loss(logits, labels):
    return F.cross_entropy(logits, labels)


def forgetting_loss(logits, labels, bnd):
    return F.relu(bnd - F.cross_entropy(logits, labels))


def main():
    device = Config.DEVICE
    logger.info(f"✅ Device: {device}")

    model = get_model().to(device)
    model = load_base_model_weights(model, Config.TRAIN.model_path(), device)

    if USE_LORA:
        enable_lora_training(model)

    logger.info("📦 Loading data...")
    train_f, train_r, val_f, val_r = get_dataloaders()

    if None in [train_f, train_r, val_f, val_r]:
        logger.error("❌ One or more DataLoaders failed to initialize.")
        return

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=Config.FORGET.LR,
                      weight_decay=Config.FORGET.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

    forget_cycle = cycle(train_f)

    best_hmean = -1
    patience = 0
    best_path = os.path.join(Config.FORGET.OUT_DIR, "best_model.pth")
    resume_path = os.path.join(Config.FORGET.OUT_DIR, "forget_resume.json")
    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)

    for epoch in range(1, Config.FORGET.EPOCHS + 1):
        model.train()
        loop = tqdm(train_r, desc=f"[Epoch {epoch}] Training", unit="batch")

        for xr, yr in loop:
            xf, yf = next(forget_cycle)
            xr, yr = xr.to(device), yr.to(device)
            xf, yf = xf.to(device), yf.to(device)

            loss_r = retention_loss(model(xr), yr)
            loss_f = forgetting_loss(model(xf), yf, Config.FORGET.BND)
            loss = loss_r + Config.FORGET.BETA * loss_f

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(Ret=f"{loss_r.item():.4f}", For=f"{loss_f.item():.4f}")

        scheduler.step()

        acc_r = compute_accuracy(model, val_r, device)
        acc_f = compute_accuracy(model, val_f, device)
        hmean = compute_hmean(acc_r, acc_f)

        logger.info(f"🧪 Epoch {epoch} — Retain Acc: {acc_r:.2f}% | Forget Acc: {acc_f:.2f}% | H-mean: {hmean:.2f}%")

        if hmean > best_hmean:
            best_hmean = hmean
            patience = 0
            torch.save(model.state_dict(), best_path)
            logger.info(f"💾 New best model saved at {best_path}")
        else:
            patience += 1
            logger.info(f"⏳ No improvement ({patience}/10)")

        with open(resume_path, "w") as f:
            json.dump({"epoch": epoch, "hmean": hmean, "model_path": best_path}, f, indent=4)

        if patience >= 10:
            logger.warning("🛑 Early stopping triggered.")
            break

    logger.info("✅ Forgetting phase complete.")


if __name__ == "__main__":
    main()
