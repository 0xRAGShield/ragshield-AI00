from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/trocr-base-handwritten",
    local_dir=r"D:\OCR_Models\hf_cache\microsoft__trocr-base-handwritten",
)

print("Download completed.")
