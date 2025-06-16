import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import logging
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------- Dataset Definition -------------------- #

class ImageDataset(Dataset):
    def __init__(self, data_path, transform=None, class_names=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []

        self.class_to_idx = {cls_name: int(cls_name.split("_")[0]) for cls_name in class_names}
        self.present_classes = sorted([int(cls.split("_")[0]) for cls in class_names])

        logger.info(f"Class-to-index mapping used: {self.class_to_idx}")
        logger.info(f"Present classes: {self.present_classes}")
        self._load_data()

    def _load_data(self):
        for class_name in self.class_to_idx:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, filename)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            size = Config.IMAGE_SIZE if isinstance(Config.IMAGE_SIZE, tuple) else (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            image = Image.new("RGB", size)
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageDatasetContiguous(Dataset):
    def __init__(self, data_path, transform=None, class_names=None):
        self.data_path = data_path
        self.transform = transform
        self.samples = []

        self.original_classes = sorted([int(cls.split("_")[0]) for cls in class_names])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_names, key=lambda x: int(x.split("_")[0])))}
        self.idx_to_original = {idx: int(cls_name.split("_")[0]) for cls_name, idx in self.class_to_idx.items()}

        logger.info(f"Class-to-contiguous-index mapping: {self.class_to_idx}")
        logger.info(f"Index-to-original-class mapping: {self.idx_to_original}")
        self._load_data()

    def _load_data(self):
        for class_name in self.class_to_idx:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for filename in os.listdir(class_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_path, filename)
                    label = self.class_to_idx[class_name]
                    self.samples.append((img_path, label))
        logger.info(f"Loaded {len(self.samples)} samples from {len(self.class_to_idx)} classes")

    def get_original_class(self, predicted_idx):
        return self.idx_to_original.get(predicted_idx, -1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            size = Config.IMAGE_SIZE if isinstance(Config.IMAGE_SIZE, tuple) else (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            image = Image.new("RGB", size)
        if self.transform:
            image = self.transform(image)
        return image, label


# -------------------- Transforms -------------------- #

def get_transforms(mode):
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.MEAN, std=Config.STD)
        ])


# -------------------- Dynamic Loader -------------------- #

def get_dynamic_loader(data_path, class_range, mode='train', batch_size=None, shuffle=None,
                       num_workers=None, pin_memory=None, drop_last=False, use_original_labels=True):
    batch_size = batch_size or Config.TRAIN.BATCH_SIZE
    shuffle = shuffle if shuffle is not None else (mode == "train")
    num_workers = num_workers if num_workers is not None else Config.NUM_WORKERS
    pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY

    all_dirs = os.listdir(data_path)

    if Config.DATA_SET.lower() == "cifar100":
        selected_classes = [cls for cls in all_dirs if cls.split("_")[0].isdigit() and int(cls.split("_")[0]) in class_range]
        selected_classes = sorted(selected_classes, key=lambda x: int(x.split("_")[0]))

        dataset_class = ImageDatasetContiguous
        logger.info("Loading CIFAR-100 formatted data...")
    else:
        selected_classes = [cls for cls in all_dirs if cls.isdigit() and int(cls) in class_range]
        selected_classes = sorted(selected_classes, key=lambda x: int(x))

        dataset_class = ImageDataset if use_original_labels else ImageDatasetContiguous
        logger.info("Loading ImageNet formatted data...")

    if not selected_classes:
        logger.warning("No classes found for the given range.")
        return None

    logger.info(f"Selected class folders: {selected_classes}")

    dataset = dataset_class(
        data_path=data_path,
        transform=get_transforms(mode),
        class_names=selected_classes
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
