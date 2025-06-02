from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


class ImageDataset(Dataset):
    def __init__(self, filenames: list[str], transform=to_tensor):
        self.filenames = filenames
        self.targets = [
            self._get_target_from_filename(filename) for filename in self.filenames
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        filename = self.filenames[index]
        target = self.targets[index]

        image = Image.open(filename)
        input_image = self.transform(image)

        return input_image, target

    def _get_target_from_filename(self, filename: str) -> int:
        """
        Get target for the given filename.

        0 indicates larvae and 1 indicates non larvae.
        """
        return int(str(filename).split("/")[-1].split("_")[0]) - 1


class ImageDataLoadBuilder:
    def __init__(
        self,
        train_perc: float = 0.50,
        valid_perc: float = 0.20,
        train_transform=to_tensor,
        valid_transform=to_tensor,
        test_transform=to_tensor,
        batchsize: int = 32,
    ):
        self._train_perc = train_perc
        self._valid_perc = valid_perc
        self._train_transform = train_transform
        self._valid_transform = valid_transform
        self._test_transform = test_transform
        self._batchsize = batchsize
        self._data_dir = "../images/larvae"

    def get_tvt_splited_filenames(self) -> tuple[list[str], list[str], list[str]]:
        filenames = glob(
            self._data_dir + "/*.png"
        )  # it returns a list of image filenames
        np.random.shuffle(filenames)

        num_train_samples = int(len(filenames) * self._train_perc)
        num_valid_samples = int(len(filenames) * self._valid_perc)

        train_filenames = filenames[:num_train_samples]
        valid_filenames = filenames[
            num_train_samples : num_train_samples + num_valid_samples
        ]
        test_filenames = filenames[num_train_samples + num_valid_samples :]

        return train_filenames, valid_filenames, test_filenames

    def get_tvt_splited_datasets(
        self,
    ) -> tuple[ImageDataset, ImageDataset, ImageDataset]:
        train_filenames, valid_filenames, test_filenames = (
            self.get_tvt_splited_filenames()
        )

        train_dataset = ImageDataset(train_filenames, self._train_transform)
        valid_dataset = ImageDataset(valid_filenames, self._valid_transform)
        test_dataset = ImageDataset(test_filenames, self._test_transform)

        return train_dataset, valid_dataset, test_dataset

    def get_tvt_splited_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, valid_dataset, test_dataset = self.get_tvt_splited_datasets()

        train_loader = DataLoader(
            train_dataset,
            batch_size=self._batchsize,
            shuffle=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self._batchsize,
            shuffle=True,
        )
        test_loader = DataLoader(test_dataset, batch_size=self._batchsize, shuffle=True)

        return train_loader, valid_loader, test_loader


def visualize_example(dataset: ImageDataset):
    def image_tensor_to_np(image_tensor: torch.Tensor) -> np.ndarray:
        def redefine_range(image: np.ndarray) -> np.ndarray:
            return 255 * (image - np.min(image)) / (np.max(image) - np.min(image))

        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = redefine_range(image_np)
        return image_np.astype("uint8")

    example_index = np.random.randint(0, len(dataset) - 1)
    image_tensor, target = dataset[example_index]

    image_np = image_tensor_to_np(image_tensor)
    class_str = "larvae" if not target else "non-larvae"

    plt.imshow(image_np)
    plt.title(class_str)
    plt.show()
