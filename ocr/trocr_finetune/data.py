#PyTorch dataset and collator for TrOCR line images

from __future__ import annotations

from typing import Sequence

from ocr.trocr_client import pad_line_keep_aspect
from ocr.trocr_finetune.dataset import SampleRecord, open_sample_image


class TrocrLineDataset:
    def __init__(
        self,
        records: Sequence[SampleRecord],
        pad_to_square: bool = True,
    ) -> None:
        self.records = list(records)
        self.pad_to_square = pad_to_square

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = open_sample_image(record)
        if self.pad_to_square:
            image = pad_line_keep_aspect(image)
        elif image.mode != "RGB":
            image = image.convert("RGB")
        return {
            "image": image,
            "text": record.text,
            "sample_id": record.sample_id,
            "image_name": record.image_name,
            "image_path": record.image_path,
        }


class TrocrCollator:
    def __init__(self, processor, max_target_length: int) -> None:
        self.processor = processor
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict]) -> dict:
        images = [item["image"] for item in batch]
        texts = [item["text"] for item in batch]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values
        tokenized = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )
        labels = tokenized.input_ids.clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if "attention_mask" in tokenized:
            labels[tokenized.attention_mask == 0] = -100
        return {"pixel_values": pixel_values, "labels": labels}
