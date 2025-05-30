# LCM Text-to-Image Generator

A Gradio-based web application for generating images using Latent Consistency Models (LCM) with Stable Diffusion XL. This implementation automatically detects and utilizes GPU if available, falling back to CPU if necessary.

## Features

- 🎨 High-quality image generation using LCM and SDXL
- ⚡ Fast inference with minimal steps (4 steps by default)
- 🔄 Automatic device detection (GPU/CPU)
- 🎯 Adjustable parameters:
  - Number of inference steps
  - Guidance scale
  - Random seed for reproducibility
- 🌐 User-friendly web interface

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Gradio web interface:
```bash
python lcm_gradio_app.py
```

2. Open your web browser and navigate to `http://localhost:7860`

3. Enter your prompt and adjust the parameters:
   - **Prompt**: Describe the image you want to generate
   - **Number of Steps**: Control the number of inference steps (1-10)
   - **Guidance Scale**: Adjust the influence of the prompt (1.0-20.0)
   - **Seed**: Set a random seed for reproducible results

4. Click "Generate Image" to create your image

## Example Prompts

- "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
- "A serene landscape with mountains and a lake at sunset, photorealistic"
- "A futuristic cityscape with flying cars and neon lights, cinematic"

## Notes

- The first run will download the necessary models, which may take some time depending on your internet connection
- GPU usage is automatically detected and enabled if available
- For optimal performance, a CUDA-capable GPU is recommended

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Latent Consistency Models](https://huggingface.co/latent-consistency/lcm-sdxl)
- [Gradio](https://gradio.app/) 