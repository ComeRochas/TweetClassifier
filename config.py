from dataclasses import dataclass

@dataclass
class TrainingConfig:
    transformer_name: str = "cardiffnlp/twitter-xlm-roberta-base"
    max_length: int = 160
    batch_size: int = 16
    num_epochs: int = 4
    lr_transformer: float = 2e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = 42
    freeze_transformer: bool = False

def get_default_config() -> TrainingConfig:
    return TrainingConfig()
