import os
import json
import torch
import logging
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.config import Config
from src.codes.data import get_val_loader
from src.pkgs.gs.vit_pytorch_face.vit_face import ViTClassifier
from src.codes.train import get_model  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_best_model(model, model_path, device):
    if os.path.exists(model_path):
        logger.info(f"🔍 Loading best model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            logger.info("✅ Model weights loaded.")
        else:
            model.load_state_dict(checkpoint)
            logger.warning("⚠️ Loaded model without full checkpoint info.")
    else:
        raise FileNotFoundError(f"No saved model found at {model_path}")
    return model


def evaluate_with_classwise(model, loader, device, num_classes):
    model.eval()
    all_preds = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            for label, pred in zip(labels, preds):
                class_total[int(label)] += 1
                if label == pred:
                    class_correct[int(label)] += 1

    overall_acc = accuracy_score(all_labels, all_preds) * 100
    class_acc = {
        str(cls): round(100 * class_correct[cls] / class_total[cls], 2) if class_total[cls] > 0 else 0.0
        for cls in range(num_classes)
    }

    return overall_acc, class_acc


def save_accuracy_report(overall_acc, class_acc, out_dir):
    acc_report = {
        "overall_accuracy": round(overall_acc, 2),
        "classwise_accuracy": class_acc
    }
    acc_path = os.path.join(out_dir, "acc.json")
    with open(acc_path, "w") as f:
        json.dump(acc_report, f, indent=4)
    logger.info(f"📄 Accuracy report saved to {acc_path}")


def main():
    device = Config.DEVICE
    val_loader = get_val_loader()
    num_classes = 40

    model = get_model().to(device)
    model = load_best_model(model, Config.TRAIN.model_path(), device)

    logger.info("📊 Evaluating best model...")
    overall_acc, class_acc = evaluate_with_classwise(model, val_loader, device, num_classes)

    logger.info(f"🏁 Overall Accuracy: {overall_acc:.2f}%")
    for cls, acc in class_acc.items():
        logger.info(f"Class {cls}: {acc:.2f}%")

    save_accuracy_report(overall_acc, class_acc, Config.TRAIN.OUT_DIR)


if __name__ == "__main__":
    main()
