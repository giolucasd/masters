from typing import Tuple

import torch
from torch import Tensor, nn

from .modules import CBAM, FireBlock, InceptionBlock, ResidualBlock


class HomogeneousCNNClassifier(nn.Module):
    """
    Homogeneous CNN classifier supporting advanced block types:
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
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
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
                FireBlock(in_channels, out_channels),
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
        """
        Initializes weights for Conv2d and Linear layers.
        - Uses Kaiming initialization for Conv2d.
        - Uses normal initialization for Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
