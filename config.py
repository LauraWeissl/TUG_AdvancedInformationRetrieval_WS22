from dataclasses import dataclass

@dataclass()
class Config:
    learning_rate: int = 1e-05
    epochs: int = 5
    batch_size: int = 4
