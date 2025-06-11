import math
import torch
from functools import partial
from src.pkgs.cl.backbone.vit_cllora import VisionTransformer

def apply_lora(vit_model, tuning_config):
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

    state_dict = vit_model.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}
    model.load_state_dict(state_dict, strict=False)

    if not getattr(tuning_config, "use_lora", True):
        print("🚫 LoRA disabled — using full fine-tuning.")
        for name, param in model.named_parameters():
            param.requires_grad = True
        return model

    for name, param in model.named_parameters():
        param.requires_grad = False

    if tuning_config.task_type == "gs":
        print("✅ Activating GS-LoRA (FFN)")
        for block in model.blocks:
            if hasattr(block, "ffn_lora_fc1") and block.ffn_lora_fc1 is not None:
                for param in block.ffn_lora_fc1.parameters():
                    param.requires_grad = True
            if hasattr(block, "ffn_lora_fc2") and block.ffn_lora_fc2 is not None:
                for param in block.ffn_lora_fc2.parameters():
                    param.requires_grad = True

    elif tuning_config.task_type == "cl":
        print("✅ Activating CL-LoRA (MSA)")
        for adapter in model.cur_adapter:
            for module in adapter:
                if hasattr(module, "lora_A") and module.lora_A is not None:
                    module.lora_A.weight.requires_grad = True
                if hasattr(module, "lora_B") and module.lora_B is not None:
                    module.lora_B.weight.requires_grad = True

    else:
        print(f"⚠️ Unknown task_type '{tuning_config.task_type}' — no adapters activated.")

    return model
