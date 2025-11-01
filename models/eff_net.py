import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
_HAS_WEIGHTS_ENUM = True


DATASET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "CIFAR10": (3, 32, 32),
    "CIFAR-10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "CIFAR-100": (3, 32, 32),
    "MNIST": (1, 28, 28),
    "FEMNIST": (1, 28, 28),
}


def _replace_batchnorm_with_groupnorm(module: nn.Module, num_groups: int = 8) -> None:
    """
    Recursively replaces all nn.BatchNorm2d layers with nn.GroupNorm.
    A default of num_groups=8 is used, which is compatible with EfficientNet-B0's
    channel sizes (e.g., 16, 24, 32, 40, 80, 112, 192, 320, 1280).
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            # Get properties from the BatchNorm layer
            num_channels = child.num_features
            eps = child.eps
            affine = child.affine
            
            # Create the new GroupNorm layer
            # We use a fixed num_groups=8, which divides all channel
            # sizes in EfficientNet-B0.
            gn = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=eps, affine=affine)
            
            # Replace the layer
            setattr(module, name, gn)
        
        else:
            # Recurse into submodules
            _replace_batchnorm_with_groupnorm(child, num_groups)


def _load_efficientnet_b0(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    Loads an EfficientNet-B0 model, modifying it for a specific
    number of input channels and classes, and for small image inputs.
    """
    if pretrained and in_channels != 3:
        # Note: Pretrained EfficientNet weights require 3-channel inputs.
        # If in_channels is not 3, the first conv layer will be
        # randomly initialized, even if pretrained=True.
        pass

    # Set weights to DEFAULT if pretrained=True, else set to None for from-scratch training
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = efficientnet_b0(weights=weights)

    # --- NEW: Replace all BatchNorm layers with GroupNorm for FL ---
    _replace_batchnorm_with_groupnorm(model)
    # --- End new code ---

    # --- 1. Modify the first conv layer for in_channels and small images ---
    # Original: model.features[0][0] = Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # We change stride=(2, 2) to stride=(1, 1) for better performance on 32x32 images.
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
        # Copy the original pretrained weights. This works as the kernel size is the same.
        new_conv1.weight.data.copy_(original_conv1.weight.data)

    model.features[0][0] = new_conv1

    # --- 2. Modify the classifier head for num_classes ---
    # Original: model.classifier = Sequential(
    #   (0): Dropout(p=0.2, inplace=True)
    #   (1): Linear(in_features=1280, out_features=1000)
    # )
    in_features = model.classifier[1].in_features # Get 1280
    model.classifier = nn.Linear(in_features, num_classes)

    return model


class EfficientNetB0Classifier(nn.Module):
    """
    Wraps torchvision EfficientNet-B0 model to accept flattened inputs.
    This version is modified to run natively on 32x32 or 28x28 images.
    """

    def __init__(
        self,
        arch: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        # --- Compatibility Check ---
        if arch != "efficientnet_b0":
            raise ValueError(
                f"This EfficientNetB0Classifier only supports 'efficientnet_b0', but got arch='{arch}'"
            )
        # ---------------------------

        self.input_shape = input_shape
        in_channels = input_shape[0]

        # --- Resize transform removed ---
        # The _load_efficientnet_b0 function modifies the model to
        # run natively on small images, so no resize is needed.

        # Load the EfficientNet-B0 model
        backbone = _load_efficientnet_b0(
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
        
        # --- Resize logic removed ---
        # The model now runs on the native input resolution (e.g., 32x32).
            
        return self.backbone(x)


def infer_image_shape(num_inputs: int, dataset: Optional[str] = None) -> Tuple[int, int, int]:
    """Guess channel/height/width from dataset name or flattened dimension."""
    if dataset is None:
        raise ValueError("Dataset must be provided to infer image shape for EfficientNet models")

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


