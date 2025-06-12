from typing import List, Optional, Tuple

import torch
from torch import nn


def conv_block(
    in_channels: int,
    out_channels: int,
    stride: int = 2,
    kernel_size: int = 3,
    use_batchnorm: bool = True,
) -> nn.Sequential:
    """
    Creates a convolutional block with optional BatchNorm and fixed ReLU + MaxPool.
    """
    layers = [
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=not use_batchnorm,
        )
    ]

    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))

    layers.extend(
        [
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=1),
        ]
    )

    return nn.Sequential(*layers)


def mlp_block(
    in_features: int,
    out_features: int,
    use_dropout: bool = True,
    dropout_prob: float = 0.2,
) -> nn.Sequential:
    """
    Creates an MLP block: Linear + ReLU (+ Dropout if enabled).
    """
    layers = [
        nn.Linear(in_features, out_features),
        nn.ReLU(),
    ]

    if use_dropout:
        layers.append(nn.Dropout(dropout_prob))

    return nn.Sequential(*layers)


class ConvNet(nn.Module):
    """
    A configurable convolutional neural network for classification tasks.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        use_dropout: bool = True,
        use_batchnorm: bool = True,
        conv_channels: Optional[List[int]] = None,
        mlp_layers: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the ConvNet.

        Args:
            input_shape: Tuple with (C, H, W) of the input image.
            num_classes: Number of output classes.
            use_dropout: Whether to use Dropout in the classifier.
            use_batchnorm: Whether to use BatchNorm in conv blocks.
            conv_channels: List of convolutional channel sizes.
            mlp_layers: List of MLP hidden layer sizes.
        """
        super().__init__()

        in_channels = input_shape[0]
        conv_channels = conv_channels or [6, 12, 24]
        mlp_layers = mlp_layers or [16]

        # Feature extractor
        layers = []
        for out_channels in conv_channels:
            layers.append(
                conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=2,
                    use_batchnorm=use_batchnorm,
                )
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # Compute flattened feature size
        fc_in_features = self._get_flattened_size(input_shape)

        # MLP classifier
        mlp_blocks = []
        for hidden in mlp_layers:
            mlp_blocks.append(
                mlp_block(
                    fc_in_features,
                    hidden,
                    use_dropout=use_dropout,
                )
            )
            fc_in_features = hidden

        mlp_blocks.append(nn.Linear(fc_in_features, num_classes))
        self.classifier = nn.Sequential(*mlp_blocks)

    def _get_flattened_size(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Computes the number of features after flattening the output from the feature extractor.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.features(dummy_input)
            return output.view(1, -1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
