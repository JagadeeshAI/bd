import os
import warnings
import logging
import torch

# Suppress warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)

# Local imports
from src.model.vit import get_vit_model
from src.model.apply_lora import apply_lora

# Utility: Count parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    percent = 100 * trainable / total if total > 0 else 0
    return total, trainable, percent

# Step 1: Load ViT base model
vit = get_vit_model(name="vit_base_patch16_224", num_classes=100)

# Step 2: Define LoRA / tuning configuration
class Args:
    def __init__(self):
        self.task_type = "gs"        # or "cl"
        self.use_lora = True         # Required for both modes
        self.ffn_adapt = True        # ✅ Required for GS-LoRA to work!
        self.vpt_on = False
        self.vpt_num = 0
        self.msa = [1, 0, 1]
        self.general_pos = [0, 1, 2, 3, 4, 5]
        self.specfic_pos = [6, 7, 8, 9, 10, 11]
        self.use_distillation = False
        self.use_block_weight = False
        self.ffn_num = 24
        self.ffn_adapter_init_option = "lora"
        self.ffn_adapter_scalar = "1.0"
        self.ffn_adapter_layernorm_option = "in"
        self.d_model = 768
        self.msa_adapt = True
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args = Args()

vit_lora = apply_lora(vit, args)

# print("\n🔍 LoRA-related Parameters:")
# for name, param in vit_lora.named_parameters():
#     if "lora" in name:
#         print(f"{name:60} | Shape: {param.shape} | {'Trainable' if param.requires_grad else 'Frozen'}")

total, trainable, percent = count_parameters(vit_lora)
print(f"\nTotal Parameters      : {total:,}")
print(f"Trainable Parameters  : {trainable:,}")
print(f"Trainable % of Total  : {percent:.4f}%")
