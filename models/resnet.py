import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn


from torchvision.models import ResNet, resnet18
from torchvision.models import ResNet18_Weights
_HAS_WEIGHTS_ENUM = True



DATASET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "CIFAR10": (3, 32, 32),
    "CIFAR-10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "CIFAR-100": (3, 32, 32),
    "MNIST": (1, 28, 28),
    "FEMNIST": (1, 28, 28),
}


def _load_resnet18(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> ResNet:
    if pretrained and in_channels != 3:
        raise ValueError(
            f"Pretrained ResNet18 weights require 3-channel inputs (got in_channels={in_channels})"
        )

    # --- THIS IS THE FIX ---
    # Set weights to DEFAULT if pretrained=True, else set to None for from-scratch training
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    # --- END FIX ---
    
    # Adjust first convolution for small images/grayscale support
    conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    if pretrained and in_channels == 3:
        # copy pretrained weights (identical shape)
        conv1.weight.data.copy_(model.conv1.weight.data)
    elif not pretrained:
        # If not pretrained, we are training from scratch, so the new conv1 is fine as is.
        # If pretrained=True and in_channels=1 (e.g., MNIST), we can't copy weights,
        # so this new conv1 layer will be trained from scratch (which is expected).
        pass 
        
    model.conv1 = conv1
    model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    
    return model


class ResNetClassifier(nn.Module):
    """Wraps torchvision ResNet18 model to accept flattened inputs."""

    def __init__(
        self,
        arch: str,  # <-- Re-added for backward compatibility
        input_shape: Tuple[int, int, int],
        num_classes: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        
        # --- Compatibility Check ---
        if arch != "resnet18":
            raise ValueError(
                f"This ResNetClassifier implementation only supports 'resnet18', but got arch='{arch}'"
            )
        # ---------------------------

        self.input_shape = input_shape
        in_channels = input_shape[0]

        # Directly load ResNet18, correctly passing the 'pretrained' flag
        backbone = _load_resnet18(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)

        self.backbone = backbone

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.view(x.size(0), *self.input_shape)
        if x.dim() == 4:
            return x
        raise ValueError(f"Unexpected input dimensions: {x.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._reshape(x)
        return self.backbone(x)


def infer_image_shape(num_inputs: int, dataset: Optional[str] = None) -> Tuple[int, int, int]:
    """Guess channel/height/width from dataset name or flattened dimension."""
    if dataset is None:
        raise ValueError("Dataset must be provided to infer image shape for ResNet models")

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