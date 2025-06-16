import os
import json
import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score

from src.config import Config
from src.codes.data import get_dynamic_loader
from src.model.vit import get_vit_model
from src.model.apply_lora import apply_lora

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Args:
    def __init__(self):
        self.task_type = "gs"
        self.use_lora = True
        self.ffn_adapt = False
        self.vpt_on = False
        self.vpt_num = 0
        self.msa = [1, 0, 1]
        self.general_pos = [0, 1, 2, 3, 4, 5]
        self.specfic_pos = [6, 7, 8, 9, 10, 11]
        self.use_distillation = True
        self.use_block_weight = True
        self.ffn_num = 8
        self.ffn_adapter_init_option = "lora"
        self.ffn_adapter_scalar = "1.0"
        self.ffn_adapter_layernorm_option = "in"
        self.d_model = 768
        self.msa_adapt = True
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    vit = get_vit_model(name="vit_base_patch16_224", num_classes=100)
    args = Args()
    model = apply_lora(vit, args, use_pretrained=False)
    return model

def evaluate(model, loader, device, num_classes):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_batches = 0

    all_preds, all_labels = [], []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    tqdm_loader = tqdm(loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for images, labels in tqdm_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # compute Cross-Entropy loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_batches += 1

            # show batch loss in progress bar
            tqdm_loader.set_postfix(loss=f"{loss.item():.4f}")

            # compute predictions
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

            # update per-class counts
            for label, pred in zip(labels, preds):
                label = int(label)
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    overall_acc = accuracy_score(all_labels, all_preds) * 100
    class_acc = {
        f"class_{cls}": round(100 * class_correct.get(cls, 0) / class_total.get(cls, 1), 2)
        for cls in range(num_classes)
    }

    return round(avg_loss, 4), round(overall_acc, 2), class_acc

def save_report(overall_acc, class_acc, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "overall_accuracy": overall_acc,
        "classwise_accuracy": class_acc,
        "total_classes": len(class_acc)
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)
    logger.info(f"📄 Report saved to {output_path}")

def main():
    device = Config.DEVICE
    logger.info(f"🖥️ Device: {device}")

    model = get_model().to(device)

    # Load best model checkpoint
    model_path = "results/checkpoints/cifer100_best.pth"
    if not os.path.exists(model_path):
        logger.error(f"🚫 Model not found at: {model_path}")
        return

    logger.info(f"📦 Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)

    # Prepare validation data loader
    val_loader = get_dynamic_loader(
        data_path=Config.FULL_VAL_DATA_PATH,
        class_range=range(0, 100),
        mode="val"
    )

    # Evaluate: get avg loss, overall accuracy, and class-wise accuracy
    avg_loss, overall_acc, class_acc = evaluate(model, val_loader, device, num_classes=100)
    logger.info(f"🔢 Avg CE Loss: {avg_loss:.4f} | 🎯 Overall Accuracy: {overall_acc:.2f}%")

    # Save results
    output_path = os.path.join(Config.TRAIN.OUT_DIR, "acc_base_cifer100.json")
    save_report(overall_acc, class_acc, output_path)

if __name__ == "__main__":
    main()
