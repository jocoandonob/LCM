import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, LCMScheduler

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def generate_image(prompt, num_steps=4, guidance_scale=8.0, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        "latent-consistency/lcm-sdxl",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    )
    
    # Load pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        unet=unet,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    
    # Set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Generate image
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

# Create Gradio interface
with gr.Blocks(title="LCM Text-to-Image Generator") as demo:
    gr.Markdown("# LCM Text-to-Image Generator")
    gr.Markdown("Generate images using Latent Consistency Models (LCM) with Stable Diffusion XL")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Enter your prompt here...",
                lines=3
            )
            with gr.Row():
                num_steps = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Number of Steps"
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=8.0,
                    step=0.5,
                    label="Guidance Scale"
                )
            seed = gr.Number(
                value=0,
                label="Seed",
                precision=0
            )
            generate_btn = gr.Button("Generate Image")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image")
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, num_steps, guidance_scale, seed],
        outputs=output_image
    )

if __name__ == "__main__":
    demo.launch() 