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


def _load_resnet18(num_classes: int, in_channels: int = 3, pretrained: bool = True) -> ResNet:
    if pretrained and in_channels != 3:
        raise ValueError(
            f"Pretrained ResNet18 weights require 3-channel inputs (got in_channels={in_channels})"
        )


    weights = ResNet18_Weights.DEFAULT  
    model = resnet18(weights=weights)

    
    # Adjust first convolution for small images/grayscale support
    conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    if pretrained and in_channels == 3:
        # copy pretrained weights (identical shape)
        conv1.weight.data.copy_(model.conv1.weight.data)
    model.conv1 = conv1
    model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    
    return model


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class _CifarResNet(nn.Module):
    def __init__(self, num_blocks: Tuple[int, int, int], num_classes: int, in_channels: int) -> None:
        super().__init__()
        self.in_planes = 16

        self.conv1 = _conv3x3(in_channels, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(_CifarBasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return torch.flatten(out, 1)


def _load_resnet20(num_classes: int, in_channels: int = 3) -> nn.Module:
    """Construct the CIFAR-style ResNet-20 network."""
    return _CifarResNet((3, 3, 3), num_classes=num_classes, in_channels=in_channels)


class ResNetClassifier(nn.Module):
    """Wraps torchvision ResNet models to accept flattened inputs."""

    def __init__(
        self,
        arch: str,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        in_channels = input_shape[0]

        if arch == "resnet18":
            backbone = _load_resnet18(num_classes=num_classes, in_channels=in_channels, pretrained=pretrained)
        elif arch == "resnet20":
            if input_shape[1:] != (32, 32):
                raise ValueError("ResNet-20 is defined for 32x32 images (CIFAR-style)")
            backbone = _load_resnet20(num_classes=num_classes, in_channels=in_channels)
        else:  # pragma: no cover - safeguarded by get_net
            raise ValueError(f"Unsupported ResNet architecture: {arch}")

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
    """
    def getPLR(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape(x)
        layers = self.backbone
        if hasattr(layers, "layer4"):
            x = layers.conv1(x)
            x = layers.bn1(x)
            x = layers.relu(x)
            x = layers.maxpool(x)
            x = layers.layer1(x)
            x = layers.layer2(x)
            x = layers.layer3(x)
            x = layers.layer4(x)
            x = layers.avgpool(x)
            return torch.flatten(x, 1)
        if hasattr(layers, "extract_features"):
            return layers.extract_features(x)
        raise AttributeError("Backbone does not expose feature extraction pathway")
    
    """


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