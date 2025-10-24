import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models.resnet import ResNet18_Weights # Keep for type consistency if mixing
_HAS_WEIGHTS_ENUM = True


DATASET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "CIFAR10": (3, 32, 32),
    "CIFAR-10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "CIFAR-100": (3, 32, 32),
    "MNIST": (1, 28, 28),
    "FEMNIST": (1, 28, 28),
}


def _load_mobilenet_v3_small(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    Loads a MobileNetV3-Small model, modifying it for small images (e.g., CIFAR/MNIST)
    and a specific number of classes.
    """
    if pretrained and in_channels != 3:
        raise ValueError(
            f"Pretrained MobileNetV3-Small weights require 3-channel inputs (got in_channels={in_channels})"
        )

    # Set weights to DEFAULT if pretrained=True, else set to None for from-scratch training
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # --- Modify for small images / grayscale support ---

    # 1. Adjust first convolution layer
    # Original: model.features[0][0] = Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # We change stride=(2, 2) to stride=(1, 1) to downsample less, which is better for 32x32 images.
    # We also adjust in_channels.
    original_conv1 = model.features[0][0]
    new_conv1 = nn.Conv2d(
        in_channels,
        original_conv1.out_channels,
        kernel_size=3,
        stride=1,  # Changed from 2
        padding=1,
        bias=False
    )

    if pretrained and in_channels == 3:
        # Copy weights. This works because kernel size (3x3) is the same.
        # The ResNet18 example had a bug here (7x7 vs 3x3), but for MobileNetV3 it's correct.
        new_conv1.weight.data.copy_(original_conv1.weight.data)
    elif not pretrained:
        # If not pretrained, we are training from scratch, so the new conv1 is fine as is.
        pass

    model.features[0][0] = new_conv1

    # 2. Adjust or remove initial downsampling
    # The ResNet18 code did `model.maxpool = nn.Identity()`.
    # The equivalent here is to remove the stride from the *first* bottleneck,
    # which is `model.features[1]`. The stride is in its first depthwise conv.
    # Original: model.features[1].block[0][0].stride = (2, 2)
    model.features[1].block[0][0].stride = (1, 1) # Changed from 2

    # 3. Adjust the final classifier
    # Original: model.classifier = Sequential(
    #   (0): Linear(in_features=576, out_features=1024, bias=True)
    #   (1): Hardswish()
    #   (2): Dropout(p=0.2, inplace=True)
    #   (3): Linear(in_features=1024, out_features=1000, bias=True)
    # )
    # We replace it with a single Linear layer, just like in the ResNet18 example.
    in_features = model.classifier[0].in_features # Get 576
    model.classifier = nn.Linear(in_features, num_classes)

    return model


class MobileNetV3SmallClassifier(nn.Module):
    """Wraps torchvision MobileNetV3-Small model to accept flattened inputs."""

    def __init__(
        self,
        arch: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # --- Compatibility Check ---
        if arch != "mobilenet_v3_small":
            raise ValueError(
                f"This MobileNetV3SmallClassifier only supports 'mobilenet_v3_small', but got arch='{arch}'"
            )
        # ---------------------------

        self.input_shape = input_shape
        in_channels = input_shape[0]

        # Directly load MobileNetV3-Small, correctly passing the 'pretrained' flag
        backbone = _load_mobilenet_v3_small(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained
        )

        self.backbone = backbone

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes a flattened 1D input tensor into a 4D image batch."""
        if x.dim() == 2:
            # Input is (batch_size, flat_features)
            return x.view(x.size(0), *self.input_shape)
        if x.dim() == 4:
            # Input is already (batch_size, C, H, W)
            return x
        raise ValueError(f"Unexpected input dimensions: {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._reshape(x)
        return self.backbone(x)


def infer_image_shape(num_inputs: int, dataset: Optional[str] = None) -> Tuple[int, int, int]:
    """Guess channel/height/width from dataset name or flattened dimension."""
    if dataset is None:
        raise ValueError("Dataset must be provided to infer image shape for ResNet/MobileNet models")

    key = dataset.upper()
    if key not in DATASET_SHAPES:
        raise ValueError(f"Unsupported dataset '{dataset}'. Supported: {sorted(set(DATASET_SHAPES))}")

    expected_shape = DATASET_SHAPES[key]
    channels, height, width = expected_shape
    flattened = channels * height * width
    if num_inputs != flattened:
        raise ValueError(
            f"Input dimension mismatch for {dataset}. Expected flattened size {flattened}, got {num_inputs}."
        )
    return expected_shape