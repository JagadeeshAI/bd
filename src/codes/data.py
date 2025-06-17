# import os
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import logging
# from src.config import Config

# class ImageDataset(Dataset):
#     def __init__(self, data_path, transform=None, class_names=None):
#         self.data_path = data_path
#         self.transform = transform
#         self.samples = []

#         self.class_to_idx = {cls_name: int(cls_name.split("_")[0]) for cls_name in class_names}
#         self.present_classes = sorted([int(cls.split("_")[0]) for cls in class_names])

#         self._load_data()

#     def _load_data(self):
#         for class_name in self.class_to_idx:
#             class_path = os.path.join(self.data_path, class_name)
#             if not os.path.isdir(class_path):
#                 continue
#             for filename in os.listdir(class_path):
#                 if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#                     img_path = os.path.join(class_path, filename)
#                     label = self.class_to_idx[class_name]
#                     self.samples.append((img_path, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             size = Config.IMAGE_SIZE if isinstance(Config.IMAGE_SIZE, tuple) else (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
#             image = Image.new("RGB", size)
#         if self.transform:
#             image = self.transform(image)
#         return image, label


# class ImageDatasetContiguous(Dataset):
#     def __init__(self, data_path, transform=None, class_names=None):
#         self.data_path = data_path
#         self.transform = transform
#         self.samples = []

#         self.original_classes = sorted([int(cls.split("_")[0]) for cls in class_names])
#         self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_names, key=lambda x: int(x.split("_")[0])))}
#         self.idx_to_original = {idx: int(cls_name.split("_")[0]) for cls_name, idx in self.class_to_idx.items()}

#         self._load_data()

#     def _load_data(self):
#         for class_name in self.class_to_idx:
#             class_path = os.path.join(self.data_path, class_name)
#             if not os.path.isdir(class_path):
#                 continue
#             for filename in os.listdir(class_path):
#                 if filename.lower().endswith((".png", ".jpg", ".jpeg")):
#                     img_path = os.path.join(class_path, filename)
#                     label = self.class_to_idx[class_name]
#                     self.samples.append((img_path, label))

#     def get_original_class(self, predicted_idx):
#         return self.idx_to_original.get(predicted_idx, -1)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         try:
#             image = Image.open(img_path).convert("RGB")
#         except Exception as e:
#             size = Config.IMAGE_SIZE if isinstance(Config.IMAGE_SIZE, tuple) else (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
#             image = Image.new("RGB", size)
#         if self.transform:
#             image = self.transform(image)
#         return image, label


# # -------------------- Transforms -------------------- #

# def get_transforms(mode):
#     if mode == "train":
#         return transforms.Compose([
#             transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomRotation(10),
#             transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=Config.MEAN, std=Config.STD)
#         ])
#     else:
#         return transforms.Compose([
#             transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=Config.MEAN, std=Config.STD)
#         ])


# # -------------------- Dynamic Loader -------------------- #

# def get_dynamic_loader(data_path, class_range, mode='train', batch_size=None, shuffle=None,
#                        num_workers=None, pin_memory=None, drop_last=False, use_original_labels=True):
#     batch_size = batch_size or Config.TRAIN.BATCH_SIZE
#     shuffle = shuffle if shuffle is not None else (mode == "train")
#     num_workers = num_workers if num_workers is not None else Config.NUM_WORKERS
#     pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY

#     all_dirs = os.listdir(data_path)

#     if Config.DATA_SET.lower() == "cifar100":
#         selected_classes = [cls for cls in all_dirs if cls.split("_")[0].isdigit() and int(cls.split("_")[0]) in class_range]
#         selected_classes = sorted(selected_classes, key=lambda x: int(x.split("_")[0]))

#         dataset_class = ImageDatasetContiguous
#     else:
#         selected_classes = [cls for cls in all_dirs if cls.isdigit() and int(cls) in class_range]
#         selected_classes = sorted(selected_classes, key=lambda x: int(x))

#         dataset_class = ImageDataset if use_original_labels else ImageDatasetContiguous

#     if not selected_classes:
#         return None


#     dataset = dataset_class(
#         data_path=data_path,
#         transform=get_transforms(mode),
#         class_names=selected_classes
#     )

#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         drop_last=drop_last
#     )



import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.config import Config


class FlexibleImageDataset(Dataset):
    def __init__(self, data_path, transform=None, class_names=None, contiguous_labels=False):
        self.data_path = data_path
        self.transform = transform
        self.samples = []
        self.contiguous_labels = contiguous_labels

        # Original labels from folder names (e.g., 0_apple → 0)
        self.original_labels = sorted([int(cls.split("_")[0]) for cls in class_names])
        self.class_names = sorted(class_names, key=lambda x: int(x.split("_")[0]))

        if contiguous_labels:
            # Map to 0...N-1
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}
            self.idx_to_original = {idx: int(cls.split("_")[0]) for idx, cls in enumerate(self.class_names)}
        else:
            # Keep original labels (e.g., 43 stays 43)
            self.class_to_idx = {cls: int(cls.split("_")[0]) for cls in self.class_names}

        self._load_data()

    def _load_data(self):
        for class_name in self.class_to_idx:
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.isdir(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(class_path, fname)
                    label = self.class_to_idx[class_name]
                    self.samples.append((path, label))

    def get_original_class(self, predicted_idx):
        if self.contiguous_labels:
            return self.idx_to_original.get(predicted_idx, -1)
        return predicted_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except:
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



def get_dynamic_loader(data_path, class_range, mode='train', batch_size=None, shuffle=None,
                       num_workers=None, pin_memory=None, drop_last=False, use_original_labels=True):
    batch_size = batch_size or Config.TRAIN.BATCH_SIZE
    shuffle = shuffle if shuffle is not None else (mode == "train")
    num_workers = num_workers if num_workers is not None else Config.NUM_WORKERS
    pin_memory = pin_memory if pin_memory is not None else Config.PIN_MEMORY

    all_dirs = os.listdir(data_path)

    if Config.DATA_SET.lower() == "cifar100":
        selected_classes = [
            cls for cls in all_dirs
            if cls.split("_")[0].isdigit() and int(cls.split("_")[0]) in class_range
        ]
        selected_classes = sorted(selected_classes, key=lambda x: int(x.split("_")[0]))

        dataset = FlexibleImageDataset(
            data_path=data_path,
            transform=get_transforms(mode),
            class_names=selected_classes,
            contiguous_labels=not use_original_labels
        )
    else:
        selected_classes = [
            cls for cls in all_dirs if cls.isdigit() and int(cls) in class_range
        ]
        selected_classes = sorted(selected_classes, key=lambda x: int(x))

        dataset = FlexibleImageDataset(
            data_path=data_path,
            transform=get_transforms(mode),
            class_names=selected_classes,
            contiguous_labels=not use_original_labels
        )

    if not selected_classes:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
