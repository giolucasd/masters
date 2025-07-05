from typing import Literal

import torch
from torch import nn
from torchvision import models as torchmodels


class PretrainedCNNClassifier(nn.Module):
    """
    CNN classifier that uses a lightweight pretrained backbone from torchvision.
    Designed for binary classification and optimized for CPU usage.
    """

    def __init__(
        self,
        num_classes: int = 1,
        backbone: Literal["mobilenet_v2", "squeezenet1_0", "resnet18"] = "mobilenet_v2",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        Initializes the model with a chosen pretrained backbone.

        Args:
            num_classes: Number of output classes (default 1 for binary classification).
            backbone: Backbone architecture to use: 'mobilenet_v2', 'squeezenet1_0', or 'resnet18'.
            device: The device to load the model to (CPU or CUDA).
        """
        super().__init__()
        self.backbone_name = backbone
        self.device = device

        self.features, in_features = self._load_backbone(backbone)
        self.classifier = self._make_classifier(in_features, num_classes)

        self.to(device)  # Ensure the entire model is moved to the specified device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor and classifier.
        """
        x = self.features(x)

        if self.backbone_name == "squeezenet1_0":
            x = x.flatten(start_dim=1)
        else:
            x = x.mean([2, 3])  # Global average pooling for other backbones

        x = self.classifier(x)

        return x

    def freeze_backbone(self):
        """
        Freezes the backbone feature extractor parameters.
        """
        for param in self.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the backbone feature extractor parameters.
        """
        for param in self.features.parameters():
            param.requires_grad = True

    def _load_backbone(self, backbone: str):
        """
        Loads the specified pretrained backbone and removes the classification head.

        Returns:
            A tuple of (feature_extractor, num_output_features)
        """
        if backbone == "mobilenet_v2":
            model = torchmodels.mobilenet_v2(
                weights=torchmodels.MobileNet_V2_Weights.DEFAULT
            )
            features = model.features
            out_features = model.last_channel

        elif backbone == "squeezenet1_0":
            model = torchmodels.squeezenet1_0(
                weights=torchmodels.SqueezeNet1_0_Weights.DEFAULT
            )
            features = nn.Sequential(
                model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            out_features = 512

        elif backbone == "resnet18":
            model = torchmodels.resnet18(weights=torchmodels.ResNet18_Weights.DEFAULT)
            modules = list(model.children())[:-2]
            features = nn.Sequential(*modules)
            out_features = model.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone '{backbone}'.")

        self.set_relu_to_non_inplace(features)

        return features.to(self.device), out_features

    def set_relu_to_non_inplace(self, module):
        for child in module.children():
            if isinstance(child, torch.nn.ReLU):
                child.inplace = False
            else:
                self.set_relu_to_non_inplace(child)

    def _make_classifier(self, in_features: int, num_classes: int) -> nn.Sequential:
        """
        Builds the classifier head.
        """
        return nn.Sequential(
            nn.Linear(in_features, 126),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(126, num_classes),
        )
