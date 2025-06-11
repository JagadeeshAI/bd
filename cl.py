from src.model.vit import get_vit_model
from src.model.apply_lora import apply_cl_lora
import torch

# Count total and trainable parameters
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, 100 * trainable / total

# Step 1: Get plain ViT
vit = get_vit_model(name="vit_base_patch16_224", num_classes=100)

# Step 2: Define CL-LoRA config
class Args:
    def __init__(self):
        self.ffn_adapt = True
        self.vpt_on = False
        self.vpt_num = 0
        self.msa = [1, 0, 1]
        self.general_pos = [0, 1, 2, 3, 4, 5]
        self.specfic_pos = [6, 7, 8, 9, 10, 11]
        self.use_distillation = False
        self.use_block_weight = False
        self.ffn_num = 8
        self.ffn_adapter_init_option = "lora"
        self.ffn_adapter_scalar = "1.0"
        self.ffn_adapter_layernorm_option = "in"
        self.d_model = 768
        self.msa_adapt = True
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ✅ Add this



args = Args()

# Step 3: Inject CL-LoRA
vit_lora = apply_cl_lora(vit, args)

# Step 4: Count and print params
total, trainable, percent = count_parameters(vit_lora)
print(f"Total Parameters      : {total:,}")
print(f"Trainable Parameters  : {trainable:,}")
print(f"Trainable % of Total  : {percent:.4f}%")
