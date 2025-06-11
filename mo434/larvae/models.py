import torch
from torch import nn


def conv_block(nchannels_in, nchannels_out, stride_val=2, kernel_size=(3, 3)):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=nchannels_in,
            out_channels=nchannels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=nchannels_out),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=nchannels_out,
            out_channels=nchannels_out,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        ),
        nn.BatchNorm2d(num_features=nchannels_out),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=kernel_size, stride=stride_val, padding=1),
    )


class ConvNet(nn.Module):
    def __init__(self, input_shape, nclasses):
        super().__init__()

        # feature extractor
        nchannels_in = input_shape[0]
        nfilters = (8, 16, 32)
        stride = 2
        self.features = nn.Sequential(
            conv_block(nchannels_in, nfilters[0], stride),
            conv_block(nfilters[0], nfilters[1], stride),
            conv_block(nfilters[1], nfilters[2], stride),
        )

        # classifier
        dim_reduction_factor = stride ** len(nfilters)
        classifier_in_features = (
            (input_shape[1] // dim_reduction_factor)
            * (input_shape[2] // dim_reduction_factor)
            * nfilters[-1]
        )
        preout_features = 16
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=classifier_in_features,
                out_features=preout_features,
                bias=True,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(preout_features, nclasses),
        )

    def forward(self, x):
        # extracts features
        x = self.features(x)

        # transforms outputs into a 2D tensor
        x = torch.flatten(x, start_dim=1)

        # classifies features
        y = self.classifier(x)

        return y
