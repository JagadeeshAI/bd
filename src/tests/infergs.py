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
        use_lora=True,
        lora_rank=8 
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
    layer_lora_stats = {}

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # Enable LoRA parameters and collect stats
    for name, module in model.named_modules():
        for attr in ['lora_A', 'lora_B', 'lora_right', 'lora_down', 'lora_up']:
            if hasattr(module, attr):
                param_or_tensor = getattr(module, attr)
                layer_key = f"{name}.{attr}"
                count = 0

                if isinstance(param_or_tensor, nn.Parameter):
                    param_or_tensor.requires_grad = True
                    count = param_or_tensor.numel()
                elif isinstance(param_or_tensor, nn.Module):
                    for p in param_or_tensor.parameters():
                        p.requires_grad = True
                        count += p.numel()

                if count > 0:
                    lora_params += count
                    layer_lora_stats[layer_key] = count

    # Logging
    if lora_params == 0:
        logger.warning("⚠️ No LoRA parameters found!")
    else:
        percent_total = 100 * lora_params / total_params
        logger.info("🔢 Parameter Summary")
        logger.info(f"  • Total model parameters: {total_params:,}")
        logger.info(f"  • Total LoRA parameters:  {lora_params:,}")
        logger.info(f"  • LoRA % of total:        {percent_total:.4f}%")
        logger.info("\n🎯 Targeted LoRA Layers:")
        for layer, count in layer_lora_stats.items():
            percent = 100 * count / total_params
            logger.info(f"  • {layer}: {count:,} params ({percent:.4f}%)")



def main():
    device = Config.DEVICE
    logger.info(f"✅ Device: {device}")

    model = get_model().to(device)
    
    enable_lora_training(model)
    
    logger.info("Test completed for gslora params.")


if __name__ == "__main__":
    main()
