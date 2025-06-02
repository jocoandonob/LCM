from huggingface_hub import snapshot_download

snapshot_download("runwayml/stable-diffusion-inpainting", force_download=True)