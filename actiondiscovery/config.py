from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    input_dim: int = 2048
    hidden_dim: int = 64
    stages: int = 4
    layers: int = 10
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    epochs: int = 160
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 15
    cosine_epochs: int = 180
    minimum_lr: float = 5e-5
    smoothing_weight: float = 0.15
    smoothing_clip: float = 16.0
    contrastive_weight: float = 0.1
    temperature: float = 0.4
    negatives: int = 5


@dataclass
class DiscoveryConfig:
    alpha: float = 0.001
    hierarchy_levels: int = 30
    max_components: int = 20
    gmm_regularization: float = 1e-6


@dataclass
class Config:
    protocol: str
    dataset_folder: str
    splits: int
    sampling_rate: int = 1
    feature_layout: str = "DT"
    normalize_features: bool = False
    seed: int = 0
    ignore_labels: list[str] = field(default_factory=list)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    @classmethod
    def from_dict(cls, values: dict) -> "Config":
        values = dict(values)
        for name, kind in [
            ("model", ModelConfig),
            ("training", TrainingConfig),
            ("discovery", DiscoveryConfig),
        ]:
            if name in values:
                values[name] = kind(**values[name])
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.feature_layout not in {"DT", "TD"}:
            raise ValueError("feature_layout must be DT or TD")
        if any(
            name in {"", ".", ".."} or Path(name).name != name
            for name in [self.protocol, self.dataset_folder]
        ):
            raise ValueError("protocol and dataset_folder must be directory names")
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in [
                self.splits,
                self.sampling_rate,
                self.model.input_dim,
                self.model.hidden_dim,
                self.model.layers,
                self.model.stages,
                self.training.epochs,
                self.training.warmup_epochs,
                self.training.cosine_epochs,
                self.training.negatives,
                self.discovery.hierarchy_levels,
                self.discovery.max_components,
            ]
        ):
            raise ValueError("dimensions, counts, and epochs must be integers")
        for section in [self.model, self.training, self.discovery]:
            for item in fields(section):
                value = getattr(section, item.name)
                if not isinstance(value, (float, int)) or not float(
                    "-inf"
                ) < value < float("inf"):
                    raise ValueError(f"{item.name} must be a finite number")
        if self.splits < 1 or self.sampling_rate < 1:
            raise ValueError("splits and sampling_rate must be positive")
        if (
            min(
                self.model.input_dim,
                self.model.hidden_dim,
                self.model.layers,
                self.model.stages,
            )
            < 1
        ):
            raise ValueError("model dimensions must be positive")
        if not 0 <= self.model.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.training.epochs < 1 or self.training.temperature <= 0:
            raise ValueError("epochs and temperature must be positive")
        if self.training.learning_rate <= 0 or self.training.minimum_lr < 0:
            raise ValueError(
                "learning rates must be nonnegative, with initial rate > 0"
            )
        if self.training.smoothing_clip <= 0:
            raise ValueError("smoothing_clip must be positive")
        if (
            self.training.warmup_epochs < 0
            or self.training.cosine_epochs <= self.training.warmup_epochs
        ):
            raise ValueError("cosine_epochs must exceed warmup_epochs >= 0")
        if (
            min(
                self.training.weight_decay,
                self.training.smoothing_weight,
                self.training.contrastive_weight,
                self.discovery.alpha,
            )
            < 0
        ):
            raise ValueError("loss weights, weight decay, and alpha cannot be negative")
        if (
            min(
                self.discovery.hierarchy_levels,
                self.discovery.max_components,
                self.training.negatives,
            )
            < 1
        ):
            raise ValueError(
                "hierarchy_levels, max_components, and negatives must be positive"
            )
        if self.discovery.gmm_regularization <= 0:
            raise ValueError("GMM regularization must be positive")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path) -> Config:
    with path.open() as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("configuration must be a YAML mapping")
    return Config.from_dict(values)
