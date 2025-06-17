import os
import torch

# 🛡️ Prevent crashes from torch.mps references on non-macOS systems
if hasattr(torch, "mps"):
    delattr(torch, "mps")

class Config:
    DATA_DIR = "/media/jag/volD/cifer100/cifer"
    FULL_TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train")
    FULL_VAL_DATA_PATH = os.path.join(DATA_DIR, "val")
    DATA_SET = "cifar100"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    PIN_MEMORY = True

    IMAGE_SIZE = 224
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    class FINETUNE:
        OUT_DIR = "./results/finetune"
        MODEL_PATH = os.path.join(OUT_DIR, "43.pth")

    class FORGET:
        OUT_DIR = "./results/forget"
        EPOCHS = 10
        LR = 3e-4
        WEIGHT_DECAY = 1e-4
        BND = 15
        BETA = 0.2
        COMP_RESUME = True
        DATA_RATIO = 0.1
        TRAIN_DATA_PATH = "/home/jag/codes/VIM_lora/data/train"
        VAL_DATA_PATH = "/home/jag/codes/VIM_lora/data/val"

        @staticmethod
        def best_model_path():
            return os.path.join(Config.FORGET.OUT_DIR, "best_model.pth")

        @staticmethod
        def resume_path():
            return os.path.join(Config.FORGET.OUT_DIR, "forget_resume.json")

    class TRAIN:
        OUT_DIR = "./results/train"
        BATCH_SIZE = 32
        EPOCHS = 300
        LR = 3e-4
        WEIGHT_DECAY = 0.05
        SNAPSHOT_INTERVAL = 10
        MAX_SNAPSHOTS = 5
        RESUME = True

        @staticmethod
        def model_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "best_model.pth")

        @staticmethod
        def progress_path():
            return os.path.join(Config.TRAIN.OUT_DIR, "progress.json")

        @staticmethod
        def snapshot_path(epoch, progress_pct):
            return os.path.join(Config.TRAIN.OUT_DIR, f"snapshot_epoch{epoch}_p{progress_pct}.pth")

# Ensure necessary output directories exist
for dir_path in [
    Config.DATA_DIR,
    Config.TRAIN.OUT_DIR,
    Config.FINETUNE.OUT_DIR,
    Config.FORGET.OUT_DIR,
]:
    os.makedirs(dir_path, exist_ok=True)

print(f"✅ Device used: {Config.DEVICE}")
