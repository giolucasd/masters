from typing import Literal, Tuple

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
        input_shape: Tuple[int, int, int],
        num_classes: int = 1,
        backbone: Literal["mobilenet_v2", "squeezenet1_0", "resnet18"] = "mobilenet_v2",
        use_dropout: bool = True,
    ) -> None:
        """
        Initializes the model with a chosen pretrained backbone.

        Args:
            input_shape: Input shape as (C, H, W).
            num_classes: Number of output classes (default 1 for binary classification).
            backbone: Backbone architecture to use: 'mobilenet_v2', 'squeezenet1_0', or 'resnet18'.
            use_dropout: Whether to use dropout in the classifier head.
        """
        super().__init__()
        self.backbone_name = backbone
        self.use_dropout = use_dropout

        self.features, in_features = self._load_backbone(backbone)

        self.classifier = self._make_classifier(in_features, num_classes, use_dropout)

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
            out_features = 512  # Final conv layer channels

        elif backbone == "resnet18":
            model = torchmodels.resnet18(weights=torchmodels.ResNet18_Weights.DEFAULT)
            modules = list(model.children())[:-2]
            features = nn.Sequential(*modules)
            out_features = model.fc.in_features

        else:
            raise ValueError(f"Unsupported backbone '{backbone}'.")

        # Freeze feature extractor for small datasets
        for param in features.parameters():
            param.requires_grad = False

        return features, out_features

    def _make_classifier(
        self, in_features: int, num_classes: int, use_dropout: bool
    ) -> nn.Sequential:
        """
        Builds the classifier head.
        """
        layers = [
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.3))
        layers.append(nn.Linear(128, num_classes))
        return nn.Sequential(*layers)
