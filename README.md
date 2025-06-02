# LCM Text-to-Image Generator

A Gradio-based web application for generating images using Latent Consistency Models (LCM) with Stable Diffusion XL. This implementation automatically detects and utilizes GPU if available, falling back to CPU if necessary.

## Features

- 🎨 High-quality image generation using five LCM methods:
  - LCM-SDXL: Full model fine-tuning approach
  - LCM-LoRA: Lightweight LoRA adapter approach
  - LCM Image-to-Image: Transform existing images with LCM
  - LCM-LoRA Image-to-Image: Transform existing images with LCM-LoRA
  - LCM-LoRA Inpainting: Edit specific areas of images with LCM-LoRA
- ⚡ Fast inference with minimal steps (4 steps by default)
- 🔄 Automatic device detection (GPU/CPU)
- 🎯 Adjustable parameters:
  - Number of inference steps
  - Guidance scale
  - Random seed for reproducibility
  - Strength (for image-to-image)
- 🌐 User-friendly web interface with tabbed interface

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

3. Choose between LCM-SDXL, LCM-LoRA, LCM Image-to-Image, LCM-LoRA Image-to-Image, or LCM-LoRA Inpainting tabs:
   - **LCM-SDXL**: Uses the full fine-tuned model (default guidance scale: 8.0)
   - **LCM-LoRA**: Uses the lightweight LoRA adapter (default guidance scale: 1.0)
   - **LCM Image-to-Image**: Transform existing images (default guidance scale: 7.5)
   - **LCM-LoRA Image-to-Image**: Transform existing images with LoRA (default guidance scale: 1.0)
   - **LCM-LoRA Inpainting**: Edit specific areas of images (default guidance scale: 4.0)

4. Enter your prompt and adjust the parameters:
   - **Prompt**: Describe the image you want to generate
   - **Initial Image**: Upload an image to transform (for image-to-image and inpainting tabs)
   - **Mask Image**: Upload a mask image for inpainting (white areas will be edited)
   - **Number of Steps**: Control the number of inference steps (1-10)
   - **Guidance Scale**: Adjust the influence of the prompt
     - LCM-SDXL: 1.0-20.0 (default: 8.0)
     - LCM-LoRA: 1.0-20.0 (default: 1.0)
     - LCM Image-to-Image: 1.0-20.0 (default: 7.5)
     - LCM-LoRA Image-to-Image: 1.0-20.0 (default: 1.0)
     - LCM-LoRA Inpainting: 1.0-20.0 (default: 4.0)
   - **Strength**: Control how much to transform the initial image (0.0-1.0, image-to-image only)
     - LCM Image-to-Image: 0.0-1.0 (default: 0.5)
     - LCM-LoRA Image-to-Image: 0.0-1.0 (default: 0.6)
   - **Seed**: Set a random seed for reproducible results

5. Click "Generate Image" to create your image

## Example Prompts

- "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
- "A serene landscape with mountains and a lake at sunset, photorealistic"
- "A futuristic cityscape with flying cars and neon lights, cinematic"
- "Astronauts in a jungle, cold color palette, muted colors, detailed, 8k" (for image-to-image)
- "concept art digital painting of an elven castle, inspired by lord of the rings, highly detailed, 8k" (for inpainting)

## Notes

- The first run will download the necessary models, which may take some time depending on your internet connection
- GPU usage is automatically detected and enabled if available
- For optimal performance, a CUDA-capable GPU is recommended
- LCM-LoRA uses less memory than LCM-SDXL but may produce slightly different results
- Different guidance scales are recommended for each method:
  - LCM-SDXL works best with higher guidance scales (8.0-12.0)
  - LCM-LoRA works best with lower guidance scales (1.0-2.0)
  - LCM Image-to-Image works best with moderate guidance scales (7.0-8.0)
  - LCM-LoRA Image-to-Image works best with lower guidance scales (1.0-2.0)
  - LCM-LoRA Inpainting works best with moderate guidance scales (4.0-6.0)
- For image-to-image:
  - Lower strength values (0.3-0.5) preserve more of the original image
  - Higher strength values (0.7-0.9) allow for more dramatic transformations
  - LCM-LoRA Image-to-Image typically works better with slightly higher strength values (0.6-0.8)
- For inpainting:
  - Use white areas in the mask to indicate regions to be edited
  - Black areas in the mask will remain unchanged
  - The mask should be the same size as the input image

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable Diffusion XL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Latent Consistency Models](https://huggingface.co/latent-consistency/lcm-sdxl)
- [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdxl)
- [LCM-LoRA SDv1.5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)
- [LCM Dreamshaper](https://huggingface.co/SimianLuo/LCM_Dreamshaper_v7)
- [Dreamshaper](https://huggingface.co/Lykon/dreamshaper-7)
- [Stable Diffusion Inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
- [Gradio](https://gradio.app/) 