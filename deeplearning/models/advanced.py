import math
from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import init


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
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


class ChannelAttention(nn.Module):
    """
    Channel Attention module for CBAM.
    Applies channel-wise attention using both max and average pooled features.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        max_pooled = self.max_pool(x)
        avg_pooled = self.avg_pool(x)
        max_out = self.mlp(max_pooled)
        avg_out = self.mlp(avg_pooled)
        attn = self.sigmoid(max_out + avg_out)
        return attn


class SpatialAttention(nn.Module):
    """
    Spatial Attention module for CBAM.
    Applies spatial attention using max and average pooling along the channel axis.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        max_pooled, _ = torch.max(x, dim=1, keepdim=True)
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        pooled = torch.cat([max_pooled, avg_pooled], dim=1)
        attn = self.sigmoid(self.conv(pooled))
        return attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies channel and spatial attention to the input feature map.
    """

    def __init__(
        self, channels: int = 512, reduction: int = 16, kernel_size: int = 7
    ) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(
            channels=channels, reduction=reduction
        )
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self) -> None:
        """Initializes weights for Conv2d, BatchNorm2d, and Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out + residual


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv2(x + self.conv1(x))


class InceptionBlock(nn.Module):
    """Simplified Inception block with 1x1, 3x3, 5x5 and pooling branches."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
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
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.device = device
        self.block_type = block_type.lower()
        in_channels = input_shape[0]
        channels = [16, 32, 64, 96]

        layers = []
        for out_channels in channels:
            block = self._make_block(in_channels, out_channels)
            layers.append(block)
            in_channels = out_channels
        self.features = nn.Sequential(*layers).to(self.device)

        self.flatten_dim = self._get_flattened_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, num_classes),
        ).to(self.device)

        self._initialize_weights()

    def _make_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Builds a convolutional block based on the selected type."""
        if self.block_type == "residual":
            return nn.Sequential(
                ResidualBlock(in_channels, out_channels, kernel_size=3),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif self.block_type == "inception":
            return nn.Sequential(
                InceptionBlock(in_channels, out_channels // 4),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif self.block_type == "se":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                SEBlock(out_channels),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        elif self.block_type == "cbam":
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                CBAM(out_channels),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        else:  # 'vanilla'
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

    def _get_flattened_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Returns the number of flattened features after the feature extractor."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape, device=self.device)
            return self.features(dummy).view(1, -1).shape[1]

    def forward(self, x: Tensor) -> Tensor:
        x = x.to(self.device)
        x = self.features(x)
        x = x.flatten(start_dim=1)
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
