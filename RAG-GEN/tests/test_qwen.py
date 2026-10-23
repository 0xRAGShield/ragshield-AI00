import torch
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_PATH = r"D:\AI\Models\qwen3-vl-8b"

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

print("Loading model...")

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    max_memory={
        0: "7GiB",
        "cpu": "20GiB",
    },
    dtype="auto",
    local_files_only=True,
)

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

print("Model loaded successfully.")
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("Model device:", next(model.parameters()).device)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Say hello and confirm that you are running locally.",
            }
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = inputs.to(model.device)

print("Generating...")

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
    )

generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]

output = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print("\nModel response:")
print(output[0])