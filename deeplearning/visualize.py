from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple
from typing import OrderedDict as OrderedDictType

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader

sns.set_theme(style="whitegrid")
sns.set_context("poster")
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
plt.rcParams["axes.spines.left"] = True
plt.rcParams["axes.spines.bottom"] = True


class Visualizer:
    """
    Visualizer provides utilities for visualizing and interpreting the internal representations and activations of a neural network model.

    Args:
        model (torch.nn.Module): The neural network model to visualize.
        device (torch.device): The device (CPU or GPU) on which computations are performed.
        labels_map (dict, optional): A mapping from label indices to label names for display purposes.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        labels_map: Optional[Dict[int, str]] = None,
    ) -> None:
        """
        Initialize the Visualizer.

        Args:
            model (nn.Module): The neural network model to visualize.
            device (torch.device): The device to use for computation.
            labels_map (dict, optional): Mapping from label indices to label names.
        """
        self.model = model
        self.device = device
        self.labels_map = labels_map

    def _get_each_output(self, x: torch.Tensor) -> OrderedDictType[str, np.ndarray]:
        """
        Collects outputs from all MaxPool2d and Linear layers using forward hooks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            OrderedDict[str, np.ndarray]: Mapping from layer names to their outputs.
        """
        outputs = OrderedDict()
        hooks = []

        def save_output(name: str) -> Callable:
            def hook(module: nn.Module, input: Any, output: Any) -> None:
                if torch.is_tensor(output):
                    outputs[name] = output.detach().cpu().numpy()

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MaxPool2d, nn.Linear)):
                hooks.append(module.register_forward_hook(save_output(name)))

        with torch.no_grad():
            _ = self.model(x)

        for h in hooks:
            h.remove()

        return outputs

    def display_layer_representations(
        self,
        dataloader: DataLoader,
        reducer: Optional[Any] = None,
    ) -> None:
        """
        Visualizes the representations of each layer for a batch of data using dimensionality reduction (e.g., t-SNE).

        Args:
            dataloader (DataLoader): DataLoader providing input samples and labels.
            reducer (object, optional): Dimensionality reduction object with fit_transform method (default: TSNE).
        """
        outputs_by_layer, all_labels = self._collect_outputs_and_labels(dataloader)
        projection_by_layer = self._reduce_and_project(outputs_by_layer, reducer)
        self._plot_projections(projection_by_layer, all_labels)

    def _collect_outputs_and_labels(
        self,
        dataloader: DataLoader,
    ) -> Tuple[OrderedDictType[str, np.ndarray], np.ndarray]:
        """
        Collects outputs and labels from the dataloader.

        Args:
            dataloader (DataLoader): DataLoader providing input samples and labels.

        Returns:
            Tuple[OrderedDict[str, np.ndarray], np.ndarray]: Outputs by layer and all labels.
        """
        outputs_by_layer = None
        all_labels = None

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels_np = labels.clone().detach().cpu().numpy()
            outputs = self._get_each_output(inputs)
            if outputs_by_layer is None:
                outputs_by_layer = outputs
                all_labels = labels_np
            else:
                for layer in outputs:
                    outputs_by_layer[layer] = np.concatenate(
                        (outputs_by_layer[layer], outputs[layer]), axis=0
                    )
                all_labels = np.concatenate((all_labels, labels_np))
        return outputs_by_layer, all_labels

    def _reduce_and_project(
        self,
        outputs_by_layer: OrderedDictType[str, np.ndarray],
        reducer: Optional[Any] = None,
    ) -> OrderedDictType[str, np.ndarray]:
        """
        Reduces and projects outputs to 2D using the given reducer.

        Args:
            outputs_by_layer (OrderedDict[str, np.ndarray]): Outputs by layer.
            reducer (object, optional): Dimensionality reduction object.

        Returns:
            OrderedDict[str, np.ndarray]: 2D projections by layer.
        """
        if reducer is None:
            reducer = TSNE(perplexity=30)

        projection_by_layer = OrderedDict()
        for layer, output in outputs_by_layer.items():
            output_reshaped = output.reshape(output.shape[0], -1)
            embedded = reducer.fit_transform(output_reshaped)
            projection_by_layer[layer] = embedded
        return projection_by_layer

    def _plot_projections(
        self,
        projection_by_layer: OrderedDictType[str, np.ndarray],
        all_labels: np.ndarray,
    ) -> None:
        """
        Plots the 2D projections for each layer.

        Args:
            projection_by_layer (OrderedDict[str, np.ndarray]): 2D projections by layer.
            all_labels (np.ndarray): Corresponding labels.
        """
        label_names = np.array(["larvae", "non-larvae"])
        all_labels_named = label_names[all_labels.astype(int)]

        for layer, embedded in projection_by_layer.items():
            fig = plt.figure(figsize=(8, 8))
            sns.scatterplot(
                x=embedded[:, 0],
                y=embedded[:, 1],
                hue=all_labels_named,
                palette="tab10",
                legend="full",
                s=40,
                edgecolor="none",
            )
            plt.axis("off")
            plt.title(layer)
            plt.show()
            plt.close(fig)

    def _get_activations(self, x: torch.Tensor) -> OrderedDictType[str, torch.Tensor]:
        """
        Collects activations from all MaxPool2d and Linear layers using forward hooks.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            OrderedDict[str, torch.Tensor]: Mapping from layer names to their activations.
        """
        activations = OrderedDict()
        hooks = []

        def save_activation(name: str):
            def hook(module, input, output):
                activations[name] = output.detach()

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.MaxPool2d, nn.Linear)):
                hooks.append(module.register_forward_hook(save_activation(name)))

        with torch.no_grad():
            _ = self.model(x)

        for h in hooks:
            h.remove()

        return activations

    def _normalize_and_resize_cam(
        self, cam: np.ndarray, target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Applies ReLU, normalizes, and resizes the CAM to the target shape.

        Args:
            cam (np.ndarray): The raw CAM.
            target_shape (Tuple[int, int]): (width, height) to resize to.

        Returns:
            np.ndarray: The processed CAM.
        """
        cam = np.maximum(cam, 0)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)
        cam = cv2.resize(cam, target_shape)
        return cam

    def _register_conv_hooks(self):
        activations = OrderedDict()
        gradients = OrderedDict()
        hooks = []

        def save_activation(name: str):
            def hook(module, input, output):
                activations[name] = output

            return hook

        def save_gradient(name: str):
            def hook(module, grad_input, grad_output):
                gradients[name] = grad_output[0]

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                hooks.append(module.register_forward_hook(save_activation(name)))
                hooks.append(module.register_full_backward_hook(save_gradient(name)))
        return hooks, activations, gradients

    def _compute_cam(self, activ, grad, input_shape):
        h, w = input_shape
        if activ.dim() == 4:  # Conv2d-like
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activ).sum(dim=1, keepdim=True)
            cam = cam[0].cpu().numpy()
            cam = self._normalize_and_resize_cam(cam[0], (w, h))
            return cam
        elif activ.dim() == 2:  # Linear-like (fallback)
            weights = grad.mean(dim=1, keepdim=True)
            cam = (weights * activ).sum(dim=1, keepdim=True)
            cam = cam[0].cpu().numpy()
            cam = self._normalize_and_resize_cam(cam, (w, h))
            return cam
        else:
            raise RuntimeError("Unsupported activation shape for CAM.")

    @staticmethod
    def image_tensor_to_np(image_tensor: torch.Tensor) -> np.ndarray:
        """
        Converts a torch image tensor (C, H, W) to a uint8 numpy array (H, W, C) for visualization.
        Scales values to [0, 255].
        """
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        image_np = (
            255
            * (image_np - np.min(image_np))
            / (np.max(image_np) - np.min(image_np) + 1e-8)
        )
        return image_np.astype("uint8")

    def _display_image_with_heatmap(
        self,
        img: np.ndarray,
        heatmap: np.ndarray,
        scale: int,
    ) -> None:
        """
        Displays the original image and the heatmap overlay side by side.

        Args:
            img (np.ndarray): Input image as a numpy array (assumed uint8 RGB).
            heatmap (np.ndarray): Heatmap to overlay.
            scale (int): Scaling factor for display size.
        """
        heatmap_uint8 = np.uint8(255.0 * heatmap)
        width = int(heatmap.shape[1] * scale)
        height = int(heatmap.shape[0] * scale)
        heatmap_resized = cv2.resize(heatmap_uint8, (width, height))
        img_resized = cv2.resize(img, (width, height))

        # Create the colored heatmap
        colored_heatmap = cv2.applyColorMap(255 - heatmap_resized, cv2.COLORMAP_JET)
        overlay = np.uint8(colored_heatmap * 0.7 + img_resized * 0.3)

        # Plot side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_resized)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Heatmap Overlay")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def display_heatmap(self, x: torch.Tensor, target: int) -> None:
        """
        Generates and displays a heatmap for the last Conv2d layer.
        Args:
            x (torch.Tensor): Input image tensor (C, H, W).
        """
        true_label = "Larvae" if target == 0 else "Non-larvae"
        print(f"  Generating heatmap for {true_label}...")

        self.model.eval()
        xin = x.unsqueeze(0).to(self.device)
        hooks, activations, gradients = self._register_conv_hooks()

        logits = self.model(xin)
        pred = logits.argmax(dim=1)
        if self.labels_map:
            print(
                f"  Predicted label is {self.labels_map[pred.cpu().detach().numpy()[0]]}!"
            )

        self.model.zero_grad()
        logits[0, pred.item()].backward(retain_graph=True)

        for h in hooks:
            h.remove()

        if not activations or not gradients:
            raise RuntimeError(
                "No activations or gradients captured for Conv2d layers."
            )

        last_layer = list(activations.keys())[-1]
        activ = activations[last_layer].detach()
        grad = gradients[last_layer].detach()

        cam = self._compute_cam(activ, grad, (x.shape[1], x.shape[2]))

        img = self.image_tensor_to_np(x)
        if img.shape[2] == 1:
            img = np.stack((img.squeeze(2),) * 3, axis=-1)
        self._display_image_with_heatmap(img, cam, scale=4)
