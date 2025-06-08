from src.pkgs.transformers.src.transformers.models.vit.image_processing_vit import ViTImageProcessor
from src.pkgs.transformers.src.transformers.models.vit.modeling_vit import ViTForImageClassification

from PIL import Image
import torch

# Load image
image_path = "/home/jag/codes/VIM_lora/data/train/3/n01491361_6.jpg"  # Adjust as needed
image = Image.open(image_path).convert("RGB")

# Load processor and model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Preprocess input
inputs = processor(images=image, return_tensors="pt")

# Inference
with torch.no_grad():
    outputs = model(**inputs)

# Prediction
logits = outputs.logits
pred_class_idx = logits.argmax(-1).item()
label = model.config.id2label[pred_class_idx]

print(f"Predicted class: {label} (index: {pred_class_idx})")

