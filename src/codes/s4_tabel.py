import os
import json
import torch
import logging
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from itertools import cycle
from torch.utils.data import Subset, DataLoader
from src.config import Config
from src.codes.data import get_dynamic_loader
from src.model.vit import get_vit_model
from src.model.apply_lora import apply_lora
from src.codes.forget import retention_loss, forgetting_loss, compute_accuracy, compute_hmean
from src.tests.acc_cifer_100 import evaluate, get_model

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
  

def main():
    device = Config.DEVICE
    logger.info(f"✅ Device: {device}")

    # Define sequential forgetting tasks: 4 tasks of 20 classes each
    splits = [list(range(i*20, (i+1)*20)) for i in range(4)]

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

    full_val_loader = get_dynamic_loader(
        data_path=Config.FULL_VAL_DATA_PATH,
        class_range=range(0, 100),
        mode="val"
    )

    avg_loss, overall_acc, class_acc = evaluate(model, val_retain, device, num_classes=100)

    results = []

    for step, forget_cls in enumerate(splits, start=1):
        retain_cls = [c for c in range(100) if c not in forget_cls]
        logger.info(f"--- Task {step}: Forgetting classes {forget_cls} | Retain classes {retain_cls[:5]}... +{len(retain_cls)-5} more")

        # Load full forgetting loader then subsample
        full_forget_loader = get_dynamic_loader(
            Config.FULL_TRAIN_DATA_PATH,
            class_range=forget_cls,
            mode='train'
        )
        # Subsample only a fraction of the forgetting data
        if Config.FORGET.DATA_RATIO < 1.0:
            dataset = full_forget_loader.dataset
            total = len(dataset)
            subset_size = int(total * Config.FORGET.DATA_RATIO)
            indices = torch.randperm(total, device=device)[:subset_size]
            train_forget = DataLoader(
                Subset(dataset, indices.cpu()),
                batch_size=full_forget_loader.batch_size,
                shuffle=True,
                num_workers=full_forget_loader.num_workers,
                pin_memory=full_forget_loader.pin_memory
            )
        else:
            train_forget = full_forget_loader

        train_retain = get_dynamic_loader(
            Config.FULL_TRAIN_DATA_PATH,
            class_range=retain_cls,
            mode='train'
        )
        val_forget   = get_dynamic_loader(
            Config.FULL_VAL_DATA_PATH,
            class_range=forget_cls,
            mode='val'
        )
        val_retain   = get_dynamic_loader(
            Config.FULL_VAL_DATA_PATH,
            class_range=retain_cls,
            mode='val'
        )

        if None in [train_forget, train_retain, val_forget, val_retain]:
            logger.error("Data loaders initialization failed for this task.")
            continue

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.FORGET.LR,
            weight_decay=Config.FORGET.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

        forget_iter = cycle(train_forget)
        best_hm = -1.0
        patience = 0
        
        # Training loop for this task
        for epoch in range(1, Config.FORGET.EPOCHS + 1):
            model.train()
            loop = tqdm(train_retain, desc=f"[Task {step} | Epoch {epoch}]", unit="batch")

            for xr, yr in loop:
                xf, yf = next(forget_iter)
                xr, yr = xr.to(device), yr.to(device)
                xf, yf = xf.to(device), yf.to(device)

                logits_r = model(xr)
                logits_f = model(xf)
                loss_r = retention_loss(logits_r, yr)
                loss_f = forgetting_loss(logits_f, yf, Config.FORGET.BND)
                loss = loss_r + Config.FORGET.BETA * loss_f

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_postfix(Ret=f"{loss_r.item():.4f}", For=f"{loss_f.item():.4f}")

            scheduler.step()

            acc_r = compute_accuracy(model, val_retain, device)
            acc_f = compute_accuracy(model, val_forget, device)
            hm = compute_hmean(acc_r, acc_f)
            logger.info(f"Task {step} Epoch {epoch} — Retain: {acc_r:.2f}% | Forget: {acc_f:.2f}% | H-Mean: {hm:.2f}%")

            if hm > best_hm:
                best_hm = hm
                patience = 0
            else:
                patience += 1
                if patience >= 10:
                    logger.warning(f"Early stopping at epoch {epoch} (no H-Mean improvement in 10 epochs)")
                    break

        results.append({
            'Step': step,
            'Classes Forgotten': step * 20,
            'Retain Acc (%)': acc_r,
            'Forget Acc (%)': acc_f,
            'H-Mean (%)': hm
        })

    print("\nContinual Forgetting Results (Table S4)\n")
    header = "| Step | Classes Forgotten | Retain Acc (%) | Forget Acc (%) | H-Mean (%) |"
    sep    = "|:----:|:----------------:|:--------------:|:-------------:|:----------:|"
    print(header)
    print(sep)
    for rec in results:
        print(f"|  {rec['Step']}  |       {rec['Classes Forgotten']}       |      {rec['Retain Acc (%)']:.2f}     |      {rec['Forget Acc (%)']:.2f}     |   {rec['H-Mean (%)']:.2f}   |")

if __name__ == "__main__":
    main()
