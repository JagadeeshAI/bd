import os
import json
import torch
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from src.config import Config
from src.codes.data import get_dynamic_loader
from src.pkgs.gs.vit_pytorch_face.vit_face import ViTClassifier

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
        use_lora=USE_LORA
    )


def load_model(model, path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Model not found at: {path}")
    logger.info(f"🔍 Loading model from: {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
    logger.info("✅ Model loaded successfully.")
    return model


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    return acc


def harmonic_mean(acc_retain, acc_forget):
    forget_complement = 1 - acc_forget / 100.0
    retain_ratio = acc_retain / 100.0
    if forget_complement + retain_ratio == 0:
        return 0.0
    return 2 * forget_complement * retain_ratio / (forget_complement + retain_ratio) * 100.0


def main():
    device = Config.DEVICE
    logger.info(f"🖥️ Device: {device}")

    model = get_model().to(device)
    model_path = Config.FORGET.best_model_path()
    model = load_model(model, model_path, device)

    # Define class ranges
    forget_classes = list(range(0, 10))
    retain_classes = list(range(10, 40))

    # Load validation loaders
    val_forget_loader = get_dynamic_loader(
        data_path=Config.FORGET.VAL_DATA_PATH,
        class_range=forget_classes,
        mode='val',
        use_original_labels=True
    )

    val_retain_loader = get_dynamic_loader(
        data_path=Config.FORGET.VAL_DATA_PATH,
        class_range=retain_classes,
        mode='val',
        use_original_labels=True
    )

    if val_forget_loader is None or val_retain_loader is None:
        logger.error("❌ Failed to load one or more validation loaders.")
        return

    acc_retain = evaluate(model, val_retain_loader, device)
    acc_forget = evaluate(model, val_forget_loader, device)
    hmean = harmonic_mean(acc_retain, acc_forget)

    logger.info(f"✅ Retain Accuracy (10–39): {acc_retain:.2f}%")
    logger.info(f"✅ Forget Accuracy (0–9): {acc_forget:.2f}%")
    logger.info(f"🎯 Harmonic Mean: {hmean:.2f}%")

    # Save report
    report = {
        "retain_accuracy": round(acc_retain, 2),
        "forget_accuracy": round(acc_forget, 2),
        "harmonic_mean": round(hmean, 2),
        "retain_classes": retain_classes,
        "forget_classes": forget_classes
    }

    os.makedirs(Config.FORGET.OUT_DIR, exist_ok=True)
    report_path = os.path.join(Config.FORGET.OUT_DIR, "acc_forget.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    logger.info(f"📄 Accuracy report saved to: {report_path}")


if __name__ == "__main__":
    main()
