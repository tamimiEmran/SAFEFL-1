import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms

from torchvision.models import vit_b_16, ViT_B_16_Weights
_HAS_WEIGHTS_ENUM = True


DATASET_SHAPES: Dict[str, Tuple[int, int, int]] = {
    "CIFAR10": (3, 32, 32),
    "CIFAR-10": (3, 32, 32),
    "CIFAR100": (3, 32, 32),
    "CIFAR-100": (3, 32, 32),
    "MNIST": (1, 28, 28),
    "FEMNIST": (1, 28, 28),
}


def _load_vit_base(num_classes: int, in_channels: int = 3, pretrained: bool = False) -> nn.Module:
    """
    Loads a ViT-Base (vit_b_16) model, modifying it for a specific
    number of input channels and classes.
    """
    if pretrained and in_channels != 3:
        # Note: Pretrained ViT weights require 3-channel inputs.
        # If in_channels is not 3, the patch projection layer will be
        # randomly initialized, even if pretrained=True.
        pass

    # Set weights to DEFAULT if pretrained=True, else set to None for from-scratch training
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model = vit_b_16(weights=weights)

    hidden_dim = model.hidden_dim
    patch_size = model.patch_size # This is 16

    # --- 1. Modify the patch projection layer for in_channels ---
    # Original: model.conv_proj = Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
    original_conv_proj = model.conv_proj
    new_conv_proj = nn.Conv2d(
        in_channels,
        hidden_dim,
        kernel_size=patch_size,
        stride=patch_size
    )

    if pretrained and in_channels == 3:
        # Copy the original pretrained weights
        new_conv_proj.load_state_dict(original_conv_proj.state_dict())

    model.conv_proj = new_conv_proj

    # --- 2. Modify the classifier head for num_classes ---
    # Original: model.heads = Sequential(Linear(in_features=768, out_features=1000))
    model.heads = nn.Linear(hidden_dim, num_classes)

    return model


class ViTBaseClassifier(nn.Module):
    """
    Wraps torchvision ViT-Base model to accept flattened inputs
    and automatically resize them to 224x224.
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
        if arch != "vit_b_16":
            raise ValueError(
                f"This ViTBaseClassifier only supports 'vit_b_16', but got arch='{arch}'"
            )
        # ---------------------------

        self.input_shape = input_shape
        in_channels = input_shape[0]

        # ViT models are not convolutional and expect a fixed input size (e.g., 224x224).
        # We add a resize transform to handle smaller inputs like CIFAR (32x32).
        self.resize = transforms.Resize([224, 224], antialias=True)

        # Load the ViT-Base model, correctly passing the 'pretrained' flag
        backbone = _load_vit_base(
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
        
        # ViT-Base (vit_b_16) expects 224x224 images.
        # We resize the input batch if it's not already the correct size.
        if x.shape[2] != 224 or x.shape[3] != 224:
            x = self.resize(x)
            
        return self.backbone(x)


def infer_image_shape(num_inputs: int, dataset: Optional[str] = None) -> Tuple[int, int, int]:
    """Guess channel/height/width from dataset name or flattened dimension."""
    if dataset is None:
        raise ValueError("Dataset must be provided to infer image shape for ViT models")

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