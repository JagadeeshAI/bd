from src.pkgs.transformers.src.transformers.models.vit.image_processing_vit import ViTImageProcessor
from src.pkgs.gs.vit_pytorch_face.vit_face import ViTClassifier

from PIL import Image
import torch
import torch.nn as nn

# ------------------ Configuration ------------------ #
IMAGE_PATH = "/home/jag/codes/VIM_lora/data/train/3/n01491361_6.jpg"
IMAGE_SIZE = 224
NUM_CLASSES = 50
USE_LORA = True
LORA_RANK = 8

LABEL_MAP = {i: f"class_{i}" for i in range(NUM_CLASSES)}

# ------------------ Load and preprocess image ------------------ #
image = Image.open(IMAGE_PATH).convert("RGB")
processor = ViTImageProcessor(
    do_resize=True,
    size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    do_normalize=True
)
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"]

# ------------------ Create model ------------------ #
model = ViTClassifier(
    num_classes=NUM_CLASSES,
    image_size=IMAGE_SIZE,
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
    use_lora=True,
    lora_rank=8
)

def enable_lora_training(model):
    """Unfreezes only LoRA-related parameters in LoRA-enabled modules."""
    count = 0
    for name, module in model.named_modules():
        for attr in ['lora_A', 'lora_B', 'lora_right', 'lora_down', 'lora_up']:
            if hasattr(module, attr):
                param_or_tensor = getattr(module, attr)
                if isinstance(param_or_tensor, nn.Parameter):
                    param_or_tensor.requires_grad = True
                    print(f"✅ Unfreezing parameter: {name}.{attr}")
                    count += param_or_tensor.numel()
                elif isinstance(param_or_tensor, nn.Module):
                    for p in param_or_tensor.parameters():
                        p.requires_grad = True
                        count += p.numel()
                    print(f"✅ Unfreezing module: {name}.{attr}")
    if count == 0:
        print("⚠️ No LoRA parameters found — check if LoRA layers are being used.")
    else:
        print(f"🔓 Enabled training for {count:,} LoRA parameters.")



# ------------------ Freeze base model if using LoRA ------------------ #
if USE_LORA:
    for param in model.parameters():
        param.requires_grad = False
    enable_lora_training(model)


# ------------------ Print stats ------------------ #
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
percent = 100.0 * trainable / total
print(f"Trainable parameters: {trainable:,} / {total:,} ({percent:.2f}%)")

# ------------------ Inference ------------------ #
with torch.no_grad():
    outputs = model(pixel_values)


print("\n🔍 Checking if LoRA layers exist:")
for name, module in model.named_modules():
    if 'MergedLinear' in module.__class__.__name__ or 'lora' in name.lower():
        print(f"✅ Found LoRA module: {name} ({module.__class__.__name__})")

# ------------------ Prediction ------------------ #
pred_class_idx = outputs.argmax(-1).item()
label = LABEL_MAP[pred_class_idx]
print(f"Predicted class: {label} (index: {pred_class_idx})")
