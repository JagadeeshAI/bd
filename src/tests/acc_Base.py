import os
import json
import torch
import logging
from collections import defaultdict
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from src.config import Config
from src.codes.data import get_dynamic_loader
from src.pkgs.gs.vit_pytorch_face.vit_face import ViTClassifier

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LoRA toggle
USE_LORA = False


def get_model():
    return ViTClassifier(
        num_classes=50,  # Ensure this matches the trained model
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


def load_best_model(model, model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚫 No model found at: {model_path}")
    
    logger.info(f"🔍 Loading best model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model_state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(model_state, strict=True)
    logger.info("✅ Model weights loaded successfully.")
    return model


def evaluate(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels = [], []
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
                label_int = int(label)
                class_total[label_int] += 1
                if label == pred:
                    class_correct[label_int] += 1

    overall_acc = accuracy_score(all_labels, all_preds) * 100

    # Construct class-wise accuracy map
    class_acc = {}

    if hasattr(loader.dataset, 'class_to_idx'):
        # Try mapping contiguous index back to original ImageNet class
        idx_to_class = {
            v: int(k) if k.isdigit() else k
            for k, v in loader.dataset.class_to_idx.items()
        }

        logger.info(f"📋 Index to class mapping: {idx_to_class}")

        for idx in range(num_classes):
            original_class = idx_to_class.get(idx, f"unknown_{idx}")
            accuracy = round(100 * class_correct[idx] / class_total[idx], 2) if class_total[idx] > 0 else 0.0
            class_acc[f"class_{original_class}"] = accuracy
    else:
        for cls in range(num_classes):
            accuracy = round(100 * class_correct[cls] / class_total[cls], 2) if class_total[cls] > 0 else 0.0
            class_acc[f"class_{cls}"] = accuracy

    return overall_acc, class_acc


def save_accuracy_report(overall_acc, class_acc, label, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    acc_report = {
        "overall_accuracy": round(overall_acc, 2),
        "classwise_accuracy": class_acc,
        "evaluation_info": {
            "total_classes": len(class_acc),
            "model_output_classes": 50,
            "class_range": "0-39"
        }
    }
    report_path = os.path.join(out_dir, f"acc_{label}.json")
    with open(report_path, "w") as f:
        json.dump(acc_report, f, indent=4)
    logger.info(f"📄 Accuracy report saved: {report_path}")


def main():
    device = Config.DEVICE
    logger.info(f"🖥️ Using device: {device}")

    # Load model and weights
    model = get_model().to(device)
    model = load_best_model(model, Config.TRAIN.model_path(), device)

    # Class range: base evaluation (same as training)
    class_range = range(0, 40)

    val_loader = get_dynamic_loader(
        data_path=Config.FULL_VAL_DATA_PATH,
        class_range=class_range,
        mode="val",
        batch_size=Config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        use_original_labels=True
    )

    if val_loader is None:
        logger.error("❌ Failed to create validation DataLoader. Exiting.")
        return

    logger.info(f"📊 Validation samples: {len(val_loader.dataset)}")
    logger.info(f"📊 Classes in validation: {len(val_loader.dataset.class_to_idx)}")
    logger.info(f"📊 Class-to-Index map: {val_loader.dataset.class_to_idx}")

    expected_classes = len(class_range)
    actual_classes = len(val_loader.dataset.class_to_idx)
    if actual_classes != expected_classes:
        logger.warning(f"⚠️ Mismatch: Expected {expected_classes} classes but found {actual_classes}")

    # Evaluate
    acc, class_acc = evaluate(model, val_loader, device, num_classes=actual_classes)
    logger.info(f"🎯 Overall Accuracy: {acc:.2f}%")

    # Preview top 10 class accuracies
    logger.info("📈 Top class accuracies:")
    for cls_name, accuracy in list(class_acc.items())[:10]:
        logger.info(f"   {cls_name}: {accuracy}%")

    # Save accuracy report
    save_accuracy_report(acc, class_acc, label="base_model", out_dir=Config.TRAIN.OUT_DIR)
    logger.info("🏁 Evaluation complete.")


if __name__ == "__main__":
    main()
