from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor

sns.set_theme(style="white")
sns.set_context("notebook")
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True


class ImageDataset(Dataset):
    """
    A custom dataset that loads images and their associated class labels inferred from file names.
    """

    def __init__(
        self,
        paths: List[Path],
        transform: Callable[[Image.Image], Tensor] = to_tensor,
    ):
        self.paths = paths
        self.targets = [self._get_target_from_path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path = self.paths[index]
        target = self.targets[index]
        image = Image.open(path)
        input_image = self.transform(image)
        return input_image, target

    def _get_target_from_path(self, path: Path) -> int:
        """
        Extract binary class label from path.
        Assumes file names are formatted as '<label>_<...>.png', where label ∈ {1, 2}.
        """
        label = int(path.stem.split("_")[0])
        return label - 1


class ImageDataLoadBuilder:
    """
    Helper class to prepare train/validation/test splits and corresponding datasets/dataloaders.
    """

    def __init__(
        self,
        data_dir: Path,
        train_perc: float = 0.50,
        valid_perc: float = 0.20,
        train_transform: Callable[[Image.Image], Tensor] = to_tensor,
        valid_transform: Callable[[Image.Image], Tensor] = to_tensor,
        test_transform: Callable[[Image.Image], Tensor] = to_tensor,
        batchsize: int = 32,
    ):
        self._train_perc = train_perc
        self._valid_perc = valid_perc
        self._train_transform = train_transform
        self._valid_transform = valid_transform
        self._test_transform = test_transform
        self._batchsize = batchsize
        self._data_dir = data_dir

    def get_tvt_splited_paths(self) -> Tuple[List[Path], List[Path], List[Path]]:
        paths = sorted(self._data_dir.glob("*.png"))
        labels = [int(p.stem.split("_")[0]) - 1 for p in paths]

        # First, split into train+valid and test
        test_perc = 1 - (self._train_perc + self._valid_perc)
        tv_paths, test_paths, tv_labels, _ = train_test_split(
            paths,
            labels,
            test_size=test_perc,
            stratify=labels,
            random_state=42,
        )

        # Then split train and validation
        rel_valid_size = self._valid_perc / (self._train_perc + self._valid_perc)
        train_paths, valid_paths, _, _ = train_test_split(
            tv_paths,
            tv_labels,
            test_size=rel_valid_size,
            stratify=tv_labels,
            random_state=42,
        )

        return train_paths, valid_paths, test_paths

    def get_tvt_splited_datasets(
        self,
    ) -> Tuple[ImageDataset, ImageDataset, ImageDataset]:
        """
        Create ImageDataset instances for train, validation, and test splits.
        """
        train_paths, valid_paths, test_paths = self.get_tvt_splited_paths()

        train_ds = ImageDataset(train_paths, self._train_transform)
        valid_ds = ImageDataset(valid_paths, self._valid_transform)
        test_ds = ImageDataset(test_paths, self._test_transform)

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


def visualize_larvae_sample(dataset: ImageDataset) -> None:
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
