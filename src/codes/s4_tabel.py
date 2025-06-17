import os, json, torch, logging
from tqdm import tqdm
from itertools import cycle
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config import Config
from src.model.vit import get_vit_model
from src.model.apply_lora import apply_lora
from src.codes.data import get_dynamic_loader
from src.codes.forget import retention_loss, forgetting_loss, compute_accuracy, compute_hmean
from src.tests.acc_cifer_100 import get_model

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_epoch_stats(task_id, epoch, acc_r, acc_f, acc_o, hmean, is_best, checkpoint_path=None):
    save_path = "results/forget/forget_resume.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    stats = {}

    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            stats = json.load(f)

    if task_id not in stats:
        stats[task_id] = []

    stats[task_id].append({
        "epoch": epoch,
        "acc_r": acc_r,
        "acc_f": acc_f,
        "acc_o": acc_o,
        "hmean": hmean,
        "is_best": is_best,
        "checkpoint": checkpoint_path if is_best else None
    })

    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)


def loaders(task_json_path, task_id):
    with open(task_json_path, "r") as f:
        task_info = json.load(f)

    assert task_id in task_info, f"Task {task_id} not found."

    step = task_info[task_id]
    forget_classes = step["forget"]
    retained_classes = step["retained"]
    already_forgotten_classes = step["already_forgotten"]

    forget_ids = [int(cls.split("_")[0]) for cls in forget_classes]
    retained_ids = [int(cls.split("_")[0]) for cls in retained_classes]
    already_forgotten_ids = [int(cls.split("_")[0]) for cls in already_forgotten_classes]

    train_path = Config.FULL_TRAIN_DATA_PATH
    val_path = Config.FULL_VAL_DATA_PATH

    # Retained classes — full loader
    train_retained_loader = get_dynamic_loader(train_path, retained_ids, mode="train", use_original_labels=True)

    # Forget classes — optionally subsample
    full_forget_loader = get_dynamic_loader(train_path, forget_ids, mode="train", use_original_labels=True)

    if Config.FORGET.DATA_RATIO < 1.0:
        dataset = full_forget_loader.dataset
        total = len(dataset)
        subset_size = int(total * Config.FORGET.DATA_RATIO)
        indices = torch.randperm(total)[:subset_size]
        train_forgotten_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=full_forget_loader.batch_size,
            shuffle=True,
            num_workers=full_forget_loader.num_workers,
            pin_memory=full_forget_loader.pin_memory,
            drop_last=False
        )
    else:
        train_forgotten_loader = full_forget_loader

    return {
        "train_retained_loader": train_retained_loader,
        "train_forgotten_loader": train_forgotten_loader,
        "val_retained_loader": get_dynamic_loader(val_path, retained_ids, mode="val", use_original_labels=True),
        "val_forgotten_loader": get_dynamic_loader(val_path, forget_ids, mode="val", use_original_labels=True),
        "val_old_forget_loader": get_dynamic_loader(val_path, already_forgotten_ids, mode="val", use_original_labels=True) if already_forgotten_ids else None,
        "val_all_loader": get_dynamic_loader(val_path, list(range(100)), mode="val", use_original_labels=True)
    }


def main():
    device = Config.DEVICE
    logger.info(f"✅ Device: {device}")

    model = get_model().to(device)

    # Load base model
    model_path = "results/checkpoints/cifer100_best.pth"
    if not os.path.exists(model_path):
        logger.error(f"🚫 Model not found at: {model_path}")
        return

    logger.info(f"📦 Loading base model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)

    # Resume state
    resume_path = "results/forget/forget_resume.json"
    resume_data = {}
    if Config.FORGET.COMP_RESUME and os.path.exists(resume_path):
        logger.info("🔄 Resuming from existing forget_resume.json...")
        with open(resume_path, "r") as f:
            resume_data = json.load(f)
    else:
        logger.info("🔁 Starting from scratch (no resume or resume disabled)")

    results = []

    for step in range(1, 5):
        task_id = f"Step{step}"

        # Skip if already completed
        if task_id in resume_data and len(resume_data[task_id]) >= Config.FORGET.EPOCHS:
            logger.info(f"⏭️ Skipping {task_id}, already completed.")
            continue

        start_epoch = len(resume_data.get(task_id, [])) + 1
        logger.info(f"\n🚀 Starting {task_id} from epoch {start_epoch}")

        data = loaders("tasks.json", task_id)

        train_retain = data["train_retained_loader"]
        train_forget = data["train_forgotten_loader"]
        val_retain = data["val_retained_loader"]
        val_forget = data["val_forgotten_loader"]
        val_all = data["val_all_loader"]

        best_ckpt_path = f"results/checkpoints/{task_id}_best.pth"
        best_hm = -1.0

        if os.path.exists(best_ckpt_path):
            logger.info(f"🔁 Loading previous best model from {best_ckpt_path}")
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"], strict=True)

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=Config.FORGET.LR,
            weight_decay=Config.FORGET.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=Config.FORGET.EPOCHS)

        forget_iter = cycle(train_forget)
        patience = 0

        for epoch in range(start_epoch, Config.FORGET.EPOCHS + 1):
            model.train()
            loop = tqdm(train_retain, desc=f"[{task_id} | Epoch {epoch}]", unit="batch")

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
            acc_o = compute_accuracy(model, val_all, device)
            hm = compute_hmean(acc_r, acc_f)

            logger.info(f"{task_id} Epoch {epoch} — Retain: {acc_r:.2f}% | Forget: {acc_f:.2f}% | Overall: {acc_o:.2f}% | H-Mean: {hm:.2f}%")

            # Save log
            save_epoch_stats(task_id, epoch, acc_r, acc_f, acc_o, hm, is_best=(hm > best_hm), checkpoint_path=best_ckpt_path if hm > best_hm else None)

            if hm > best_hm:
                best_hm = hm
                patience = 0
                torch.save({"model_state": model.state_dict()}, best_ckpt_path)
            else:
                patience += 1
                if patience >= 10:
                    logger.warning(f"⏹️ Early stopping at epoch {epoch} (no H-Mean improvement in 10 epochs)")
                    break

        results.append({
            'Step': step,
            'Classes Forgotten': step * 20,
            'Retain Acc (%)': acc_r,
            'Forget Acc (%)': acc_f,
            'Overall Acc (%)': acc_o,
            'H-Mean (%)': hm
        })

    print("\n📊 Continual Forgetting Results (Table S4 format)\n")
    print("| Step | Classes Forgotten | Retain Acc (%) | Forget Acc (%) | Overall Acc (%) | H-Mean (%) |")
    print("|:----:|:-----------------:|:---------------:|:---------------:|:----------------:|:----------:|")
    for r in results:
        print(f"|  {r['Step']}  |        {r['Classes Forgotten']}         |      {r['Retain Acc (%)']:.2f}     |      {r['Forget Acc (%)']:.2f}     |      {r['Overall Acc (%)']:.2f}     |   {r['H-Mean (%)']:.2f}   |")


if __name__ == "__main__":
    main()
