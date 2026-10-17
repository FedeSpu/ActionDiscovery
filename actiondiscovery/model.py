"""MS-TCN backbone (https://github.com/yabufarha/ms-tcn) adapted.

Input: [batch, feature_dim, time]. Output logits: [stage, batch, class, time].
Embeddings: [batch, hidden_dim, time], before the final classification layer.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, channels: int, dropout: float):
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.dropout(
            self.conv_1x1(F.relu(self.conv_dilated(features)))
        )


class SingleStageModel(nn.Module):
    def __init__(self, input_dim: int, classes: int, config: ModelConfig):
        super().__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, config.hidden_dim, 1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(2**layer, config.hidden_dim, config.dropout)
                for layer in range(config.layers)
            ]
        )
        self.conv_class_pred = nn.Conv1d(config.hidden_dim, classes, 1)

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.conv_1x1(features)
        for layer in self.layers:
            embeddings = layer(embeddings)
        embeddings = embeddings * mask
        return embeddings, self.conv_class_pred(embeddings) * mask


class MultiStageModel(nn.Module):
    def __init__(self, config: ModelConfig, known_classes: int):
        super().__init__()
        classes = known_classes + 1  # 0 = UNK; 1..N = known actions.
        self.stage1 = SingleStageModel(config.input_dim, classes, config)
        self.stages = nn.ModuleList(
            [
                SingleStageModel(classes, classes, config)
                for _ in range(config.stages - 1)
            ]
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            mask = torch.ones_like(features[:, :1])
        embeddings, logits = self.stage1(features, mask)
        outputs = [logits]
        for stage in self.stages:
            embeddings, logits = stage(F.softmax(logits, dim=1), mask)
            outputs.append(logits)
        return torch.stack(outputs), embeddings
