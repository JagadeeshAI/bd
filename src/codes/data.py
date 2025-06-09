import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None, class_names=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_names))}
        self._load_data()

    def _load_data(self):
        for class_name in self.class_to_idx:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_path, filename)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            size = Config.IMAGE_SIZE if isinstance(Config.IMAGE_SIZE, tuple) else (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            image = Image.new("RGB", size)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.MEAN, std=Config.STD)
    ])


def get_class_names_in_range(data_path, start, end):
    all_classes = os.listdir(data_path)
    selected = [cls for cls in all_classes if cls.isdigit() and start <= int(cls) <= end]
    return sorted(selected, key=lambda x: int(x))


# -------------------- Loaders -------------------- #

def get_train_loader():
    class_names = get_class_names_in_range(Config.FULL_TRAIN_DATA_PATH, 0, 39)
    dataset = ImageDataset(Config.FULL_TRAIN_DATA_PATH, transform=get_train_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=True,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True)

def train_forget():
    class_names = get_class_names_in_range(Config.FULL_TRAIN_DATA_PATH, 0, 10)
    dataset = ImageDataset(Config.FULL_TRAIN_DATA_PATH, transform=get_train_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=True,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True)


def get_val_loader():
    class_names = get_class_names_in_range(Config.FULL_VAL_DATA_PATH, 0, 39)
    dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=get_val_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=False,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

def val_forget():
    class_names = get_class_names_in_range(Config.FULL_VAL_DATA_PATH, 0, 10)
    dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=get_val_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=False,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)


def get_full_train_loader():
    class_names = get_class_names_in_range(Config.FULL_TRAIN_DATA_PATH, 0, 49)
    dataset = ImageDataset(Config.FULL_TRAIN_DATA_PATH, transform=get_train_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=True,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True)


def get_full_val_loader():
    class_names = get_class_names_in_range(Config.FULL_VAL_DATA_PATH, 0, 49)
    dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=get_val_transforms(), class_names=class_names)
    return DataLoader(dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=False,
                      num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)


# ---------------- Forget Loader ---------------- #

def get_forget_loader():
    print("\n🧠 Choose a class label to forget (0–39):")
    class_names = get_class_names_in_range(Config.FULL_TRAIN_DATA_PATH, 0, 39)
    for i, name in enumerate(class_names):
        print(f"[{i}] Class: {name}")

    try:
        idx = int(input("Enter index of class to forget: ").strip())
        forget_class = class_names[idx]
    except (ValueError, IndexError):
        raise ValueError("Invalid class index. Please select a valid number between 0 and 39.")

    print(f"\n⚠️  Forgetting class: {forget_class}")

    forget_class_list = [forget_class]

    forget_train_dataset = ImageDataset(Config.FULL_TRAIN_DATA_PATH, transform=get_train_transforms(), class_names=forget_class_list)
    forget_val_dataset = ImageDataset(Config.FULL_VAL_DATA_PATH, transform=get_val_transforms(), class_names=forget_class_list)

    forget_train_loader = DataLoader(forget_train_dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=True,
                                     num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True)

    forget_val_loader = DataLoader(forget_val_dataset, batch_size=Config.TRAIN.BATCH_SIZE, shuffle=False,
                                   num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY)

    return forget_train_loader, forget_val_loader


# -------------------- Testing -------------------- #

def test_loader(loader_fn, name):
    print(f"\n🔍 Testing {name}")
    try:
        loader = loader_fn() if callable(loader_fn) else loader_fn
        print(f"✅ {name} dataset loaded: {len(loader.dataset)} samples")
        print(f"   - Number of batches: {len(loader)}")
        images, labels = next(iter(loader))
        print(f"   - Sample batch shape: images={images.shape}, labels={labels.shape}")
    except Exception as e:
        print(f"❌ Failed to load {name}: {e}")
    print("-" * 60)


def main():
    print("=" * 60)
    print("🧪 Testing All Data Loaders")
    print("=" * 60)

    test_loader(get_train_loader, "Train Loader (0–39)")
    test_loader(get_val_loader, "Validation Loader (0–39)")
    test_loader(get_full_train_loader, "Full Train Loader (0–49)")
    test_loader(get_full_val_loader, "Full Val Loader (0–49)")

    # Test forget loader
    forget_train_loader, forget_val_loader = get_forget_loader()
    test_loader(forget_train_loader, "Forget Train Loader")
    test_loader(forget_val_loader, "Forget Val Loader")

    print("=" * 60)


if __name__ == "__main__":
    main()