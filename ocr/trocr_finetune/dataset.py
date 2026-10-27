#Load RxHandBD image/transcription pairs without rewriting the source files

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from PIL import Image

from ocr.preprocessing import SUPPORTED_SUFFIXES
from ocr.trocr_finetune.config import TrocrFinetuneConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_UNCERTAIN = re.compile(r"^\?+$")
IMAGE_COLUMN_ALIASES = ("Images", "Image", "image", "filename", "file")
TEXT_COLUMN_ALIASES = ("Text", "text", "label", "transcription", "ground_truth")


@dataclass
class SampleIssue:
    code: str
    message: str
    image_name: str | None = None
    split: str | None = None


@dataclass
class SampleRecord:
    sample_id: str
    image_name: str
    text: str
    official_split: str
    split: str
    image_path: str | None
    issues: list[str] = field(default_factory=list)
    contains_arabic: bool = False
    uncertain: bool = False
    usable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetBundle:
    records: list[SampleRecord]
    issues: list[SampleIssue]
    counts: dict
    config: dict
    tokenizer_notes: list[str]

    def split_records(self, name: str, usable_only: bool = True) -> list[SampleRecord]:
        rows = [row for row in self.records if row.split == name]
        if usable_only:
            return [row for row in rows if row.usable]
        return rows

    def to_dict(self) -> dict:
        return {
            "counts": dict(self.counts),
            "config": dict(self.config),
            "tokenizer_notes": list(self.tokenizer_notes),
            "issues": [asdict(item) for item in self.issues],
            "records": [row.to_dict() for row in self.records],
        }

    def write_manifest(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination


def load_rxhandbd(config: TrocrFinetuneConfig) -> DatasetBundle:
    issues: list[SampleIssue] = []
    train_csv = resolve_project_path(config.train_csv)
    test_csv = resolve_project_path(config.test_csv)
    raw_csv = resolve_project_path(config.raw_csv)
    train_image_dir = resolve_project_path(config.train_image_dir)
    test_image_dir = resolve_project_path(config.test_image_dir)

    official_train = _read_label_csv(train_csv, "official_train", issues)
    official_test = _read_label_csv(test_csv, "official_test", issues)
    official_test_names = {row["image_name"] for row in official_test}

    if config.source == "ml":
        working = list(official_train)
        source_note = f"official Train_Label.csv; images from {train_image_dir}"
        train_lookup_dirs = [train_image_dir]
    else:
        working = _read_label_csv(raw_csv, "raw", issues)
        source_note = (
            "RxHandBD-Raw Prescription_Labels.csv; official test IDs still held out"
        )
        train_lookup_dirs = [train_image_dir, *[resolve_project_path(item) for item in config.image_dirs]]

    if not train_image_dir.is_dir():
        issues.append(
            SampleIssue(
                "missing_train_image_dir",
                f"Training image directory not found: {train_image_dir}",
                split="train",
            )
        )
    if not test_image_dir.is_dir():
        issues.append(
            SampleIssue(
                "missing_test_image_dir",
                f"Test image directory not found: {test_image_dir}",
                split="test",
            )
        )

    train_index = _index_image_dir(train_lookup_dirs)
    test_index = _index_image_dir([test_image_dir])
    seen_names: dict[str, str] = {}
    seen_pairs: set[tuple[str, str]] = set()
    records: list[SampleRecord] = []

    for row in working:
        name = row["image_name"]
        text = _optional_normalize(row["text"], config)
        split = _assign_split(name, official_test_names, config)
        path = _resolve_image(name, train_index)
        sample_issues: list[str] = []
        uncertain = _is_uncertain(text)
        arabic = bool(_ARABIC.search(text))

        if not text:
            sample_issues.append("empty_transcription")
            issues.append(
                SampleIssue("empty_transcription", "Ground-truth text is empty", name, split)
            )
        if "?" in text:
            sample_issues.append("contains_question_mark")
            issues.append(
                SampleIssue(
                    "contains_question_mark",
                    "Ground Truth contains '?'; it is kept unless drop_uncertain and the whole string is placeholders",
                    name,
                    split,
                )
            )
        if uncertain:
            sample_issues.append("uncertain_transcription")
            issues.append(
                SampleIssue(
                    "uncertain_transcription",
                    "Transcription is only question marks or equivalent placeholder",
                    name,
                    split,
                )
            )
        if arabic:
            sample_issues.append("contains_arabic")
        if path is None:
            sample_issues.append("missing_image")
            issues.append(
                SampleIssue(
                    "missing_image",
                    f"No file matching {name} under {train_image_dir}",
                    name,
                    split,
                )
            )
        if name in seen_names:
            sample_issues.append("duplicate_image_name")
            issues.append(
                SampleIssue(
                    "duplicate_image_name",
                    f"Duplicate filename already seen in {seen_names[name]}",
                    name,
                    split,
                )
            )
        pair = (name.lower(), text)
        if pair in seen_pairs:
            sample_issues.append("duplicate_image_text_pair")
        else:
            seen_pairs.add(pair)
        if name not in seen_names:
            seen_names[name] = split

        usable = (
            path is not None
            and bool(text)
            and "duplicate_image_name" not in sample_issues
            and not (config.drop_uncertain and uncertain)
            and split != "held_out_conflict"
        )
        if split == "held_out_conflict":
            sample_issues.append("leaks_into_official_test")
            issues.append(
                SampleIssue(
                    "split_leak",
                    "Image id is in the official test CSV; excluded from train/val",
                    name,
                    split,
                )
            )
            usable = False

        records.append(
            SampleRecord(
                sample_id=Path(name).stem,
                image_name=name,
                text=text,
                official_split=row["official_split"],
                split=split,
                image_path=str(path) if path else None,
                issues=sample_issues,
                contains_arabic=arabic,
                uncertain=uncertain,
                usable=usable,
            )
        )

    for row in official_test:
        name = row["image_name"]
        text = _optional_normalize(row["text"], config)
        path = _resolve_image(name, test_index)
        sample_issues: list[str] = []
        if not text:
            sample_issues.append("empty_transcription")
        if "?" in text:
            sample_issues.append("contains_question_mark")
        if _is_uncertain(text):
            sample_issues.append("uncertain_transcription")
        if path is None:
            sample_issues.append("missing_image")
            issues.append(
                SampleIssue(
                    "missing_image",
                    f"No file matching {name} under {test_image_dir}",
                    name,
                    "test",
                )
            )
        usable = path is not None and bool(text) and not (
            config.drop_uncertain and _is_uncertain(text)
        )
        records.append(
            SampleRecord(
                sample_id=Path(name).stem,
                image_name=name,
                text=text,
                official_split="official_test",
                split="test",
                image_path=str(path) if path else None,
                issues=sample_issues,
                contains_arabic=bool(_ARABIC.search(text)),
                uncertain=_is_uncertain(text),
                usable=usable,
            )
        )

    counts = {
        "listed_train_candidates": sum(1 for row in records if row.split == "train"),
        "listed_val": sum(1 for row in records if row.split == "val"),
        "listed_test": sum(1 for row in records if row.split == "test"),
        "usable_train": sum(1 for row in records if row.split == "train" and row.usable),
        "usable_val": sum(1 for row in records if row.split == "val" and row.usable),
        "usable_test": sum(1 for row in records if row.split == "test" and row.usable),
        "missing_images": sum(1 for row in records if "missing_image" in row.issues),
        "empty_transcriptions": sum(1 for row in records if "empty_transcription" in row.issues),
        "uncertain_transcriptions": sum(
            1 for row in records if "uncertain_transcription" in row.issues
        ),
        "arabic_transcriptions": sum(1 for row in records if row.contains_arabic),
        "duplicate_image_names": sum(
            1 for row in records if "duplicate_image_name" in row.issues
        ),
        "issue_events": len(issues),
        "source": source_note,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "train_image_dir": str(train_image_dir),
        "test_image_dir": str(test_image_dir),
    }
    notes = tokenizer_compatibility_notes()
    if counts["arabic_transcriptions"]:
        notes.append(
            "Some Ground Truth strings contain Arabic letters. The existing "
            "RobertaTokenizer vocabulary is English BPE (vocab_size 50265). "
            "This pipeline does not resize embeddings. Arabic would be "
            "fragmented if encoded; fine-tuning is not expected to teach "
            "reliable Arabic OCR with this tokenizer."
        )
    else:
        notes.append(
            "No Arabic-script characters were found in the RxHandBD CSV labels "
            "inspected by this loader. Labels are Latin-script medication names, "
            "digits, and punctuation."
        )
    if counts["missing_images"]:
        notes.append(
            "Some CSV filenames did not resolve to a file in the split-specific "
            "image directory. Labels are not invented for missing images."
        )

    return DatasetBundle(
        records=records,
        issues=issues,
        counts=counts,
        config=config.to_dict(),
        tokenizer_notes=notes,
    )


def open_sample_image(record: SampleRecord) -> Image.Image:
    if not record.image_path:
        raise FileNotFoundError(f"No image path for {record.image_name}")
    with Image.open(record.image_path) as opened:
        opened.load()
        return opened.convert("RGB")


def tokenizer_compatibility_notes() -> list[str]:
    return [
        "Target checkpoint: microsoft/trocr-base-handwritten "
        "(VisionEncoderDecoderModel, ViT encoder 384px, TrOCR decoder).",
        "Processor class in checkpoint config: TrOCRProcessor. "
        "Tokenizer class in tokenizer_config.json: RobertaTokenizer "
        "(name_or_path roberta-large, vocab_size 50265 from config.json).",
        "pad_token_id=1, eos/decoder_start_token_id=2, bos_token_id=0 "
        "(generation_config.json / decoder config). Unverified at runtime; "
        "taken from files on disk.",
        "This tokenizer is English byte-level BPE. Dedicated Arabic letters "
        "are not part of a handwritten-Arabic OCR vocabulary. This project "
        "does not replace the tokenizer or resize token embeddings.",
        "microsoft/trocr-base-handwritten is documented in this repository as "
        "English IAM line-level OCR. Image encoding does not imply Arabic support.",
    ]


def _read_label_csv(
    path: Path,
    official_split: str,
    issues: list[SampleIssue],
) -> list[dict]:
    if not path.is_file():
        issues.append(
            SampleIssue("missing_csv", f"Label CSV not found: {path}", split=official_split)
        )
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            issues.append(SampleIssue("empty_csv", f"{path} is empty", split=official_split))
            return []
        image_idx = _column_index(header, IMAGE_COLUMN_ALIASES, 0)
        text_idx = _column_index(header, TEXT_COLUMN_ALIASES, 1)
        for line_number, fields in enumerate(reader, start=2):
            if not fields or all(not cell.strip() for cell in fields):
                issues.append(
                    SampleIssue(
                        "malformed_row",
                        f"{path.name} line {line_number} is empty",
                        split=official_split,
                    )
                )
                continue
            image_name = fields[image_idx].strip() if image_idx < len(fields) else ""
            text = fields[text_idx] if text_idx < len(fields) else ""
            if not image_name:
                issues.append(
                    SampleIssue(
                        "malformed_row",
                        f"{path.name} line {line_number} has no image filename",
                        split=official_split,
                    )
                )
                continue
            suffix = Path(image_name).suffix.lower()
            if suffix and suffix not in SUPPORTED_SUFFIXES:
                issues.append(
                    SampleIssue(
                        "unsupported_image_suffix",
                        f"{image_name} uses suffix {suffix}",
                        image_name,
                        official_split,
                    )
                )
            rows.append(
                {
                    "image_name": image_name,
                    "text": text,
                    "official_split": official_split,
                    "source_csv": str(path),
                    "line_number": line_number,
                }
            )
    return rows


def _column_index(header: list[str], aliases: tuple[str, ...], fallback: int) -> int:
    cleaned = [cell.strip() for cell in header]
    for alias in aliases:
        if alias in cleaned:
            return cleaned.index(alias)
    return fallback


def _optional_normalize(text: str, config: TrocrFinetuneConfig) -> str:
    value = text.replace("\ufeff", "")
    if config.unicode_nfkc:
        value = unicodedata.normalize("NFKC", value)
    if config.strip_text:
        value = value.strip()
    if config.collapse_internal_whitespace:
        value = " ".join(value.split())
    return value


def _is_uncertain(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return False
    return bool(_UNCERTAIN.match(compact.replace(" ", "")))


def _assign_split(
    image_name: str,
    official_test_names: set[str],
    config: TrocrFinetuneConfig,
) -> str:
    if image_name in official_test_names:
        return "held_out_conflict"
    digest = hashlib.sha256(f"{config.seed}:{image_name.lower()}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < config.val_fraction_of_official_train:
        return "val"
    return "train"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative paths against the OCR project root, not the process CWD."""
    value = Path(path)
    if value.is_absolute():
        return value
    return (PROJECT_ROOT / value).resolve()


def _index_image_dir(directories: list[Path]) -> dict[str, Path]:
    """Map exact names, lower-case names, and lower-case stems to files.

    Directories are not mixed across train and test by the caller. File
    extensions are taken from disk rather than assumed.
    """
    index: dict[str, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if not path.is_file():
                continue
            index.setdefault(path.name, path)
            index.setdefault(path.name.lower(), path)
            index.setdefault(path.stem.lower(), path)
    return index


def _resolve_image(name: str, index: dict[str, Path]) -> Path | None:
    filename = Path(name).name
    for key in (filename, filename.lower(), Path(filename).stem.lower()):
        found = index.get(key)
        if found is not None and found.is_file():
            return found
    return None
