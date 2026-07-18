from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from transformers import RobertaModel


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """3D ResNet-18 feature extractor with a 512-dimensional output."""

    def __init__(self) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )
        layers = [BasicBlock3D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        layers.extend(BasicBlock3D(self.in_planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return torch.flatten(self.avgpool(x), 1)


class FeatureAttention(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.network(x), dim=1)
        return x * weights + x


class VLINC(nn.Module):
    def __init__(self, model_cfg: Dict, radiologic_dir: str) -> None:
        super().__init__()
        self.freeze_report_encoder = bool(model_cfg["freeze_report_encoder"])
        self.image_encoder = ResNet3D()
        self.report_encoder = RobertaModel.from_pretrained(radiologic_dir, add_pooling_layer=False)
        report_dim = int(self.report_encoder.config.hidden_size)
        configured_report_dim = int(model_cfg["report_feature_dim"])
        if report_dim != configured_report_dim:
            raise ValueError(
                f"RadioLOGIC hidden size is {report_dim}, but model.report_feature_dim is "
                f"{configured_report_dim}."
            )
        if self.freeze_report_encoder:
            for parameter in self.report_encoder.parameters():
                parameter.requires_grad = False
            self.report_encoder.eval()

        self.clinical_projection = nn.Linear(int(model_cfg["clinical_feature_dim"]), 512)
        self.mutation_projection = nn.Linear(int(model_cfg["mutation_feature_dim"]), 256)
        self.therapy_projection = nn.Linear(int(model_cfg["therapy_feature_dim"]), 128)
        self.prompt_embedding = nn.Embedding(2, int(model_cfg["prompt_embedding_dim"]))

        multimodal_dim = 2 * 512 + report_dim + 512 + 256 + 128
        self.attention = FeatureAttention(multimodal_dim, int(model_cfg["attention_hidden_dim"]))
        self.fusion = nn.Sequential(
            nn.Linear(multimodal_dim + int(model_cfg["prompt_embedding_dim"]), int(model_cfg["fusion_hidden_dim"])),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(int(model_cfg["fusion_hidden_dim"])),
            nn.Dropout(float(model_cfg["dropout"])),
            nn.Linear(int(model_cfg["fusion_hidden_dim"]), 1),
        )

    def train(self, mode: bool = True) -> "VLINC":
        super().train(mode)
        if self.freeze_report_encoder:
            self.report_encoder.eval()
        return self

    def _encode_report(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.freeze_report_encoder:
            with torch.no_grad():
                output = self.report_encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict=False,
                )[0]
        else:
            output = self.report_encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict=False,
            )[0]
        return output[:, 0]

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image1 = self.image_encoder(batch["image1"])
        image2 = self.image_encoder(batch["image2"])
        report = self._encode_report(batch)
        clinical = self.clinical_projection(batch["clinical_features"].flatten(1).float())
        mutation = self.mutation_projection(batch["mutation_features"].float())
        therapy = self.therapy_projection(batch["therapy_features"].float())
        features = self.attention(torch.cat([image1, image2, report, clinical, mutation, therapy], dim=1))
        prompt = self.prompt_embedding(batch["prompt_type"])
        return self.fusion(torch.cat([prompt, features], dim=1)).squeeze(1)
