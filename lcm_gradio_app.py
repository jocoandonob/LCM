import gradio as gr
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    LCMScheduler,
    DiffusionPipeline,
    AutoPipelineForImage2Image,
    AutoPipelineForInpainting
)
from diffusers.utils import load_image

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def generate_image_lcm_sdxl(prompt, num_steps=4, guidance_scale=8.0, seed=0):
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

def generate_image_lcm_lora(prompt, num_steps=4, guidance_scale=1.0, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16" if device == "cuda" else None,
        torch_dtype=dtype
    ).to(device)
    
    # Set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    
    # Generate image
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

def generate_image_custom_lora(prompt, num_steps=4, guidance_scale=8.0, seed=0):
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
    
    # Load custom LoRA weights with static parameters
    pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")
    
    # Generate image
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        generator=generator,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image

def generate_image2image_lcm(prompt, init_image, num_steps=4, guidance_scale=7.5, strength=0.5, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load UNet
    unet = UNet2DConditionModel.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        subfolder="unet",
        torch_dtype=dtype,
    )
    
    # Load pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "Lykon/dreamshaper-7",
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
        image=init_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator
    ).images[0]
    
    return image

def generate_image2image_lcm_lora(prompt, init_image, num_steps=4, guidance_scale=1.0, strength=0.6, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load pipeline
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "Lykon/dreamshaper-7",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
    ).to(device)
    
    # Set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    
    # Generate image
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        image=init_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        generator=generator
    ).images[0]
    
    return image

def generate_inpainting_lcm_lora(prompt, init_image, mask_image, num_steps=4, guidance_scale=4.0, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load pipeline
    pipe = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        # use_safetensors=True
    ).to(device)
    
    # Set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    
    # Generate image
    generator = torch.manual_seed(seed)
    image = pipe(
        prompt=prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    return image

def generate_image_multi_lora(prompt, num_steps=4, guidance_scale=1.0, seed=0):
    device = get_device()
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        variant="fp16" if device == "cuda" else None,
        torch_dtype=dtype
    ).to(device)
    
    # Set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Load multiple LoRA weights
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl", adapter_name="lcm")
    pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")
    
    # Set adapter weights
    pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])
    
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
    
    with gr.Tabs():
        with gr.TabItem("LCM-SDXL"):
            with gr.Row():
                with gr.Column():
                    prompt_sdxl = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    with gr.Row():
                        num_steps_sdxl = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_sdxl = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=8.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    seed_sdxl = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_sdxl = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_sdxl = gr.Image(label="Generated Image")
            
            generate_btn_sdxl.click(
                fn=generate_image_lcm_sdxl,
                inputs=[prompt_sdxl, num_steps_sdxl, guidance_scale_sdxl, seed_sdxl],
                outputs=output_image_sdxl
            )
        
        with gr.TabItem("LCM-LoRA"):
            with gr.Row():
                with gr.Column():
                    prompt_lora = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    with gr.Row():
                        num_steps_lora = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_lora = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=1.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    seed_lora = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_lora = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_lora = gr.Image(label="Generated Image")
            
            generate_btn_lora.click(
                fn=generate_image_lcm_lora,
                inputs=[prompt_lora, num_steps_lora, guidance_scale_lora, seed_lora],
                outputs=output_image_lora
            )
        
        with gr.TabItem("Custom LoRA"):
            with gr.Row():
                with gr.Column():
                    prompt_custom_lora = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    with gr.Row():
                        num_steps_custom_lora = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_custom_lora = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=8.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    seed_custom_lora = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_custom_lora = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_custom_lora = gr.Image(label="Generated Image")
            
            generate_btn_custom_lora.click(
                fn=generate_image_custom_lora,
                inputs=[
                    prompt_custom_lora,
                    num_steps_custom_lora,
                    guidance_scale_custom_lora,
                    seed_custom_lora
                ],
                outputs=output_image_custom_lora
            )
        
        with gr.TabItem("LCM Image-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt_img2img = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    init_image = gr.Image(
                        label="Initial Image",
                        type="pil"
                    )
                    with gr.Row():
                        num_steps_img2img = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_img2img = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=7.5,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.05,
                        label="Strength"
                    )
                    seed_img2img = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_img2img = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_img2img = gr.Image(label="Generated Image")
            
            generate_btn_img2img.click(
                fn=generate_image2image_lcm,
                inputs=[
                    prompt_img2img,
                    init_image,
                    num_steps_img2img,
                    guidance_scale_img2img,
                    strength,
                    seed_img2img
                ],
                outputs=output_image_img2img
            )
        
        with gr.TabItem("LCM-LoRA Image-to-Image"):
            with gr.Row():
                with gr.Column():
                    prompt_img2img_lora = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    init_image_lora = gr.Image(
                        label="Initial Image",
                        type="pil"
                    )
                    with gr.Row():
                        num_steps_img2img_lora = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_img2img_lora = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=1.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    strength_lora = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.6,
                        step=0.05,
                        label="Strength"
                    )
                    seed_img2img_lora = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_img2img_lora = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_img2img_lora = gr.Image(label="Generated Image")
            
            generate_btn_img2img_lora.click(
                fn=generate_image2image_lcm_lora,
                inputs=[
                    prompt_img2img_lora,
                    init_image_lora,
                    num_steps_img2img_lora,
                    guidance_scale_img2img_lora,
                    strength_lora,
                    seed_img2img_lora
                ],
                outputs=output_image_img2img_lora
            )
        
        with gr.TabItem("LCM-LoRA Inpainting"):
            with gr.Row():
                with gr.Column():
                    prompt_inpaint = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    init_image_inpaint = gr.Image(
                        label="Initial Image",
                        type="pil"
                    )
                    mask_image_inpaint = gr.Image(
                        label="Mask Image",
                        type="pil"
                    )
                    with gr.Row():
                        num_steps_inpaint = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_inpaint = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=4.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    seed_inpaint = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_inpaint = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_inpaint = gr.Image(label="Generated Image")
            
            generate_btn_inpaint.click(
                fn=generate_inpainting_lcm_lora,
                inputs=[
                    prompt_inpaint,
                    init_image_inpaint,
                    mask_image_inpaint,
                    num_steps_inpaint,
                    guidance_scale_inpaint,
                    seed_inpaint
                ],
                outputs=output_image_inpaint
            )
        
        with gr.TabItem("Multi-LoRA"):
            with gr.Row():
                with gr.Column():
                    prompt_multi_lora = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your prompt here...",
                        lines=3
                    )
                    with gr.Row():
                        num_steps_multi_lora = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4,
                            step=1,
                            label="Number of Steps"
                        )
                        guidance_scale_multi_lora = gr.Slider(
                            minimum=1.0,
                            maximum=20.0,
                            value=1.0,
                            step=0.5,
                            label="Guidance Scale"
                        )
                    seed_multi_lora = gr.Number(
                        value=0,
                        label="Seed",
                        precision=0
                    )
                    generate_btn_multi_lora = gr.Button("Generate Image")
                
                with gr.Column():
                    output_image_multi_lora = gr.Image(label="Generated Image")
            
            generate_btn_multi_lora.click(
                fn=generate_image_multi_lora,
                inputs=[
                    prompt_multi_lora,
                    num_steps_multi_lora,
                    guidance_scale_multi_lora,
                    seed_multi_lora
                ],
                outputs=output_image_multi_lora
            )

if __name__ == "__main__":
    demo.launch() 