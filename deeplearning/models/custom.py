from typing import Tuple

import torch
from torch import Tensor, nn


class CustomClassifier(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.device = device
        in_channels = input_shape[0]

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels, 5, kernel_size=3, stride=1, padding="same", bias=False,
            ),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ).to(self.device)

        self.branch5 = nn.Sequential(
            nn.Conv2d(
                in_channels, 5, kernel_size=5, stride=1, padding="same", bias=False,
            ),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ).to(self.device)

        self.branch7 = nn.Sequential(
            nn.Conv2d(
                in_channels, 5, kernel_size=7, stride=1, padding="same", bias=False,
            ),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ).to(self.device)

        self.skip_proj = nn.Sequential(
            nn.Conv2d(15, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
        ).to(self.device)

        self.block2 = nn.Sequential(
            nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ).to(self.device)

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ).to(self.device)

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        ).to(self.device)

        self.classifier = nn.Sequential(
            nn.Linear(14 * 14 * 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        ).to(self.device)

        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch3(x)
        x2 = self.branch5(x)
        x3 = self.branch7(x)

        x = torch.cat((x1, x2, x3), dim=1)

        skip = x

        x = self.block2(x)
        x = self.block3(x)

        for _ in range(2):
            skip = nn.functional.avg_pool2d(skip, kernel_size=3, stride=2, padding=1)
        skip = self.skip_proj(skip)

        x = x + skip
        x = self.block4(x)

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
