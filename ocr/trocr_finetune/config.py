
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ocr.config import DEFAULT_TROCR_CHECKPOINT_DIR, TROCR_MODEL_ID


@dataclass
class TrocrFinetuneConfig:
   

    dataset_root: str = "dataset"
    source: str = "ml"
    train_csv: str = "dataset/RxHandBD-ML/Train_Label.csv"
    test_csv: str = "dataset/RxHandBD-ML/Test_Label.csv"
    train_image_dir: str = "dataset/RxHandBD-ML/Train_Set"
    test_image_dir: str = "dataset/RxHandBD-ML/Test_Set"
    raw_csv: str = "dataset/RxHandBD-Raw/Prescription_Labels.csv"
    image_dirs: list[str] = field(default_factory=list)
    source_checkpoint_dir: str = str(DEFAULT_TROCR_CHECKPOINT_DIR)
    model_id: str = TROCR_MODEL_ID
    output_dir: str = "outputs/trocr_finetune/run"
    seed: int = 42
    val_fraction_of_official_train: float = 0.10
    strip_text: bool = True
    collapse_internal_whitespace: bool = False
    unicode_nfkc: bool = False
    drop_uncertain: bool = False
    pad_line_keep_aspect: bool = True
    max_target_length: int = 64
    epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    logging_steps: int = 20
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 2
    early_stopping_patience: int = 2
    resume_from_checkpoint: str | None = None
    device: str = "cpu"
    local_files_only: bool = True
    max_train_samples: int | None = None
    max_eval_samples: int | None = None

    def __post_init__(self) -> None:
        if self.source not in {"ml", "raw"}:
            raise ValueError("source must be 'ml' or 'raw'")
        if not 0.0 < self.val_fraction_of_official_train < 0.5:
            raise ValueError("val_fraction_of_official_train must be in (0, 0.5)")
        if self.max_target_length < 8:
            raise ValueError("max_target_length must be >= 8")
        if self.fp16 and self.bf16:
            raise ValueError("Set only one of fp16 or bf16")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrocrFinetuneConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {key: payload[key] for key in cls.__dataclass_fields__ if key in payload}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination
