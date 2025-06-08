import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from transformers import ViTConfig, ViTModel
from src.config import Config
from src.codes.data import (
    get_incremental_train_loader,
    get_val_loader,
    get_incremental_val_loader,
    get_full_val_loader,
)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.optim import AdamW
from transformers import ViTConfig, ViTModel
from src.config import Config
from src.codes.data import (
    get_incremental_train_loader,
    get_val_loader,
    get_incremental_val_loader,
    get_full_val_loader,
)

class LoRA(nn.Module):
    def __init__(self, dim, r=10, fixed_B=False):
        super().__init__()
        self.r = r
        self.fixed_B = fixed_B
        if fixed_B:
            B = torch.randn(r, dim)
            self.register_buffer('B', torch.linalg.qr(B.T)[0].T)
            self.A = nn.Parameter(torch.zeros(dim, r))
        else:
            self.A = nn.Parameter(torch.zeros(dim, r))
            self.B = nn.Parameter(torch.randn(r, dim) * 0.01)

    def forward(self, x):
        return x @ self.B.T @ self.A.T

class CLLoRA_ViT(nn.Module):
    def __init__(self, num_classes=50, shared_blocks=6, r=10):
        super().__init__()
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
            num_labels=num_classes
        )
        self.vit = ViTModel(config)
        self.shared_blocks = shared_blocks
        self.total_blocks = config.num_hidden_layers
        self.r = r

        self.shared_adapters = nn.ModuleList([LoRA(config.hidden_size, r=r, fixed_B=True) for _ in range(shared_blocks)])
        self.task_adapters = nn.ModuleList()
        self.block_weights = nn.ParameterList()

        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def add_task_adapters(self):
        adapters = nn.ModuleList([LoRA(768, r=self.r).to(next(self.parameters()).device) for _ in range(self.total_blocks - self.shared_blocks)])
        weights = nn.Parameter(torch.ones(self.total_blocks - self.shared_blocks).to(next(self.parameters()).device))
        self.task_adapters.append(adapters)
        self.block_weights.append(weights)

    def forward(self, x):
        out = self.vit(pixel_values=x)
        x = out.last_hidden_state

        for i in range(self.total_blocks):
            if i < self.shared_blocks:
                x = x + self.shared_adapters[i](x)
            else:
                task_idx = len(self.task_adapters) - 1
                idx = i - self.shared_blocks
                x = x + self.task_adapters[task_idx][idx](x) * self.block_weights[task_idx][idx]

        x = self.vit.layernorm(x)
        cls_token_final = x[:, 0]
        return self.classifier(cls_token_final)

    def forward_shared_only(self, x):
        out = self.vit(pixel_values=x)
        x = out.last_hidden_state

        for i in range(self.shared_blocks):
            x = x + self.shared_adapters[i](x)

        return x[:, 0]

    def forward_train(self, x, y):
        logits = self.forward(x)
        ce_loss = F.cross_entropy(logits, y)
        total_loss = ce_loss
        return logits, {'total': total_loss, 'ce': ce_loss}

def count_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params / total_params * 100


def train():
    device = Config.DEVICE
    outdir = Path(Config.TRAIN.OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    model = CLLoRA_ViT(num_classes=50).to(device)

    base_checkpoint = "results/train/base_53.pth"
    if os.path.exists(base_checkpoint):
        print(f"📦 Loading base model from {base_checkpoint}")
        checkpoint = torch.load(base_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model.add_task_adapters()

    for name, param in model.named_parameters():
        param.requires_grad = "adapters" in name

    print(f"🔧 Tunable Parameters: {count_trainable_params(model):.4f}%")

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.TRAIN.LR, weight_decay=Config.TRAIN.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.TRAIN.EPOCHS)

    train_loader = get_incremental_train_loader()
    val_loader_all = get_full_val_loader()

    for epoch in range(1, Config.TRAIN.EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, losses = model.forward_train(images, labels)
            losses['total'].backward()
            optimizer.step()
            total_loss += losses['total'].item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Loss: {avg_loss:.4f}")

        acc_all, val_loss = evaluate(model, val_loader_all, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {acc_all:.2f}%")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total_loss = 0, 0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        correct += (outputs.argmax(1) == labels).sum().item()
        total_loss += loss.item()
    return 100 * correct / len(loader.dataset), total_loss / len(loader)

if __name__ == "__main__":
    train()