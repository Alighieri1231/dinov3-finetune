import os
import zipfile
import logging
import urllib.request
from typing import Optional

import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation


VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root: str = "./data",
        year: str = "2012",
        image_set: str = "train",
        download: bool = True,
        transform: Optional[A.Compose] = None,
        use_index_label: bool = True,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
        )
        self.n_classes = 21
        self.transform = transform
        self.use_index_label = use_index_label

    @staticmethod
    def _convert_to_segmentation_mask(
        mask: np.ndarray, use_index_label: bool = True
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)),
            dtype=np.float32,
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)

        if use_index_label:
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)
        return segmentation_mask

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = self._convert_to_segmentation_mask(mask, self.use_index_label)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        mask = np.moveaxis(mask, -1, 0)
        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


class ADE20kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[A.Compose] = None,
    ):
        self.root = root
        self.split = split
        self.n_classes = 150
        self.transform = transform

        root = os.path.join(root, "ADEChallengeData2016")
        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        # Check if the dataset is already downloaded
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            self.download_and_extract_dataset()

        self.image_files = os.listdir(self.images_dir)

    def download_and_extract_dataset(self) -> None:
        dataset_url = (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )
        zip_path = os.path.join(self.root, "ADEChallengeData2016.zip")
        os.makedirs(self.root, exist_ok=True)

        logging.info("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)

        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        logging.info("Dataset extracted!")
        os.remove(zip_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) - 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


def get_dataloader(
    dataset_name: str,
    img_dim: tuple[int, int] = (490, 490),
    batch_size: int = 6,
) -> tuple[DataLoader, DataLoader]:
    assert dataset_name in ["ade20k", "voc"], "dataset name not in [ade20k, voc]"
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    if dataset_name == "voc":
        train_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="train",
            download=False,
            transform=transform,
        )
        val_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="train",
            download=False,
            transform=transform,
        )
    elif dataset_name == "ade20k":
        train_dataset = ADE20kDataset(
            root="./data",
            split="training",
            transform=transform,
        )

        val_dataset = ADE20kDataset(
            root="./data",
            split="validation",
            transform=transform,
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
