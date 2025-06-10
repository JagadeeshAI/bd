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
        num_classes=50,
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
        logger.warning("⚠️ No LoRA parameters found!")
    else:
        percent = 100 * lora_params / total_params
        logger.info(f"🔓 Training LoRA only: {lora_params:,} / {total_params:,} ({percent:.2f}%)")


def get_dataloaders():
    train_path = Config.FORGET.TRAIN_DATA_PATH
    val_path = Config.FORGET.VAL_DATA_PATH

    forget_classes = list(range(0, 10))
    retain_classes = list(range(10, 40))

    
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

    return  val_forget_loader, val_retain_loader


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

def main():
    device = Config.DEVICE
    logger.info(f"✅ Device: {device}")

    model = get_model().to(device)
    model = load_base_model_weights(model, "results/forget/53accr0accf.pth", device)

    if USE_LORA:
        enable_lora_training(model)

    logger.info("📦 Loading data...")
    val_f, val_r = get_dataloaders()

    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)


    acc_r = compute_accuracy(model, val_r, device)
    acc_f = compute_accuracy(model, val_f, device)
    hmean = compute_hmean(acc_r, acc_f)

    logger.info(f" — Retain Acc: {acc_r:.2f}% | Forget Acc: {acc_f:.2f}% | H-mean: {hmean:.2f}%")

    logger.info("✅ Forgetting phase complete.")


if __name__ == "__main__":
    main()
