from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


class ImageDataset(Dataset):
    """
    A custom dataset that loads images and their associated binary class labels
    inferred from filenames. Class label 0 indicates larvae and 1 indicates non-larvae.
    """

    def __init__(
        self,
        filenames: List[Path],
        transform: Callable[[Image.Image], Tensor] = to_tensor,
    ):
        self.filenames = filenames
        self.targets = [self._get_target_from_filename(fname) for fname in filenames]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        filename = self.filenames[index]
        target = self.targets[index]
        image = Image.open(filename)
        input_image = self.transform(image)
        return input_image, target

    def _get_target_from_filename(self, filename: Path) -> int:
        """
        Extract binary class label from filename.
        Assumes filenames are formatted as '<label>_<...>.png', where label ∈ {1, 2}.
        Returns 0 for larvae and 1 for non-larvae.
        """
        label = int(filename.stem.split("_")[0])
        return label - 1


class ImageDataLoadBuilder:
    """
    Helper class to prepare train/validation/test splits and corresponding datasets/dataloaders.
    """

    def __init__(
        self,
        train_perc: float = 0.50,
        valid_perc: float = 0.20,
        train_transform: Callable[[Image.Image], Tensor] = to_tensor,
        valid_transform: Callable[[Image.Image], Tensor] = to_tensor,
        test_transform: Callable[[Image.Image], Tensor] = to_tensor,
        batchsize: int = 32,
        data_dir: Path = Path("../images/larvae"),
    ):
        self._train_perc = train_perc
        self._valid_perc = valid_perc
        self._train_transform = train_transform
        self._valid_transform = valid_transform
        self._test_transform = test_transform
        self._batchsize = batchsize
        self._data_dir = data_dir

    def get_tvt_splited_filenames(self) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Load and split image file paths into train, validation, and test sets based on provided proportions.
        """
        filenames = sorted(self._data_dir.glob("*.png"))
        np.random.shuffle(filenames)

        total = len(filenames)
        n_train = int(total * self._train_perc)
        n_valid = int(total * self._valid_perc)

        train = filenames[:n_train]
        valid = filenames[n_train : n_train + n_valid]
        test = filenames[n_train + n_valid :]

        return train, valid, test

    def get_tvt_splited_datasets(
        self,
    ) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
        """
        Create ImageDataset instances for train, validation, and test splits.
        """
        train_files, valid_files, test_files = self.get_tvt_splited_filenames()

        train_ds = ImageDataset(train_files, self._train_transform)
        valid_ds = ImageDataset(valid_files, self._valid_transform)
        test_ds = ImageDataset(test_files, self._test_transform)

        return train_ds, valid_ds, test_ds

    def get_tvt_splited_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoader instances for train, validation, and test datasets.
        """
        train_ds, valid_ds, test_ds = self.get_tvt_splited_datasets()

        train_loader = DataLoader(train_ds, batch_size=self._batchsize, shuffle=True)
        valid_loader = DataLoader(valid_ds, batch_size=self._batchsize, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=self._batchsize, shuffle=True)

        return train_loader, valid_loader, test_loader


def visualize_example(dataset: ImageDataset) -> None:
    """
    Randomly visualize a single example from the dataset.
    """

    def image_tensor_to_np(image_tensor: Tensor) -> np.ndarray:
        image_np = image_tensor.permute(1, 2, 0).numpy()
        image_np = (
            255 * (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np))
        )
        return image_np.astype("uint8")

    index = np.random.randint(0, len(dataset))
    image_tensor, target = dataset[index]
    image_np = image_tensor_to_np(image_tensor)

    class_label = "larvae" if target == 0 else "non-larvae"
    plt.imshow(image_np)
    plt.title(class_label)
    plt.axis("off")
    plt.show()
