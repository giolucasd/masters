import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


class FlexCNNClassifier(nn.Module):
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
        Initializes the FlexCNNClassifier.

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
                self._make_conv_block(
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
                self._make_mlp_block(
                    fc_in_features,
                    hidden,
                    use_dropout=use_dropout,
                )
            )
            fc_in_features = hidden

        mlp_blocks.append(nn.Linear(fc_in_features, num_classes))
        self.classifier = nn.Sequential(*mlp_blocks)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        """
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def _get_flattened_size(self, input_shape: Tuple[int, int, int]) -> int:
        """
        Computes the number of features after flattening the output from the feature extractor.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.features(dummy_input)
            return output.view(1, -1).shape[1]

    def _make_conv_block(
        self,
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

    def _make_mlp_block(
        self,
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

    def _initialize_weights(self):
        """
        Initializes weights for Conv2d and Linear layers using standard strategies:
        - Conv2d: He initialization (normal distribution with std = sqrt(2 / fan_in))
        - Linear: Normal distribution with mean = 0 and std = 0.01
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel and Spatial attention)."""

    def __init__(
        self, channels: int, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Channel attention
        avg_out = self.channel_attention(x)
        x = x * avg_out

        # Spatial attention
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial = self.spatial_attention(torch.cat([avg_pool, max_pool], dim=1))

        return x * spatial


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.conv(x))


class InceptionBlock(nn.Module):
    """Simplified Inception block with 1x1, 3x3, 5x5 and pooling branches."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2),
        )

        self.pool_proj = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        )

        self.output_bn = nn.BatchNorm2d(4 * out_channels)

    def forward(self, x: Tensor) -> Tensor:
        outputs = [
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.pool_proj(x),
        ]
        return self.output_bn(torch.cat(outputs, dim=1))


class EnhancedCNNClassifier(nn.Module):
    """
    CNN classifier supporting advanced block types:
    'vanilla', 'residual', 'inception', 'se', and 'cbam'.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        block_type: str = "vanilla",
    ) -> None:
        super().__init__()

        self.block_type = block_type.lower()
        C, _, _ = input_shape
        channels = [8, 16, 32]
        in_channels = C

        layers = []
        for out_channels in channels:
            block = self._make_block(in_channels, out_channels)
            layers.append(block)
            in_channels = self._get_output_channels(out_channels)
        self.features = nn.Sequential(*layers)

        self.flatten_dim = self._get_flattened_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self._initialize_weights()

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Builds a convolutional block based on the selected type."""
        if self.block_type == "residual":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                ResidualBlock(out_channels),
                nn.MaxPool2d(2),
            )
        elif self.block_type == "inception":
            return nn.Sequential(
                InceptionBlock(in_channels, out_channels // 4),
                nn.MaxPool2d(2),
            )
        elif self.block_type == "se":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                SEBlock(out_channels),
                nn.MaxPool2d(2),
            )
        elif self.block_type == "cbam":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                CBAM(out_channels),
                nn.MaxPool2d(2),
            )
        else:  # 'vanilla'
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

    def _get_output_channels(self, base_channels: int) -> int:
        """Returns the number of output channels depending on the block type."""
        if self.block_type == "inception":
            return base_channels * 4
        return base_channels

    def _get_flattened_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Returns the number of flattened features after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)

    def _initialize_weights(self) -> None:
        """Initializes Conv2d and Linear layers using standard initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
