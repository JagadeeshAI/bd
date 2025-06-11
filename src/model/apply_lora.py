import math
import torch

from src.pkgs.cl.backbone.vit_cllora import VisionTransformer
from functools import partial

def apply_cl_lora(vit_model, tuning_config):
    from src.pkgs.cl.backbone.vit_cllora import VisionTransformer
    from functools import partial

    # Step 1: Instantiate CL-LoRA VisionTransformer
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        tuning_config=tuning_config,
    )

    # Step 2: Copy pretrained weights (ignore classifier head)
    state_dict = vit_model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}  # 🚨 exclude classifier
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("⚠️ Ignored keys while loading:", missing, unexpected)

    # Step 3: Freeze all but LoRA adapters
    for name, param in model.named_parameters():
        param.requires_grad = False
    for adapter in model.cur_adapter:
        for module in adapter:
            if hasattr(module, "lora_A"):
                module.lora_A.weight.requires_grad = True
            if hasattr(module, "lora_B"):
                module.lora_B.weight.requires_grad = True

    return model

