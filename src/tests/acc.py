import os
import json
import torch
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

from torch import nn
from transformers import ViTConfig, ViTForImageClassification

from src.config import Config
from src.codes.data import get_full_val_loader

def load_model():
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
    model_path = "results/train/incremental_epoch1.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found at: {model_path}")

    checkpoint = torch.load(model_path, map_location=Config.DEVICE)

    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(Config.DEVICE)
    model.eval()
    print(f"✅ Model loaded from {model_path}")
    return model

def evaluate_per_class(model, loader, device, num_classes=50):
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running Inference"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            preds = outputs.argmax(dim=1)

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            for label, pred in zip(labels.cpu(), preds.cpu()):
                class_total[int(label)] += 1
                if pred == label:
                    class_correct[int(label)] += 1

    class_acc = {}
    for cls in range(num_classes):
        correct = class_correct[cls]
        total = class_total[cls]
        acc = 100.0 * correct / total if total > 0 else 0.0
        class_acc[str(cls)] = round(acc, 2)

    overall_acc = 100.0 * total_correct / total_samples if total_samples > 0 else 0.0
    return class_acc, round(overall_acc, 2)

def save_results(results, out_path="results.json"):
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"✅ Per-class accuracy saved to {out_path}")

def main():
    device = Config.DEVICE
    print(f"✅ Device used: {device}")

    val_loader = get_full_val_loader()
    model = load_model()

    print("🚀 Evaluating model on validation set...")
    class_acc, overall_acc = evaluate_per_class(model, val_loader, device, num_classes=50)

    output_path = Path(Config.TRAIN.OUT_DIR) / "results.json"
    save_results(class_acc, output_path)

    print(f"\n📊 Overall Accuracy (classes 0–49): {overall_acc:.2f}%\n")

if __name__ == "__main__":
    main()
