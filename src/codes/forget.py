import os
import torch
from model import build_model  # your model definition file
from src.codes.data import train_forget , val_forget
from src.codes.backboneForgetting import backbone_forgetting
from src.config import Config




def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Using device: {device}")

    model = build_model(num_classes=40)  
    model.load_state_dict(torch.load("results/train/base_53.pth", map_location=device))
    print("✅ Loaded base model.")

    

    # Perform forgetting
    backbone_forgetting(
        model=model,
        device=device,
        loaders=loaders,
        task_id=1,
        config=Config,
        start_epoch=0,
        resume_path=os.path.join(Config.FORGET.OUT_DIR, "resume.json")
    )


if __name__ == "__main__":
    main()
