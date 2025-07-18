import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.wrapper import StreamDiffusionWrapper
import torch
from config import Args
from pydantic import BaseModel, Field
from PIL import Image

base_model = "stabilityai/sd-turbo"
default_prompt = "Portrait of The Joker halloween costume, face painting, with , glare pose, detailed, intricate, full of colour, cinematic lighting, trending on artstation, 8k, hyperrealistic, focused, extreme details, unreal engine 5 cinematic, masterpiece"
default_negative_prompt = ""

page_content = """<h1 class="text-3xl font-bold">StreamDiffusion</h1>
<h3 class="text-xl font-bold">Image-to-Image SD-Turbo</h3>
<p class="text-sm">Optimized for maximum FPS</p>
"""

class Pipeline:
    class Info(BaseModel):
        name: str = "StreamDiffusion img2img"
        input_mode: str = "image"
        page_content: str = page_content

    class InputParams(BaseModel):
        prompt: str = Field(
            default_prompt,
            title="Prompt",
            field="textarea",
            id="prompt",
        )
        strength: float = Field(
            0.5,
            min=0.0,
            max=1.0,
            step=0.1,
            title="Denoising Strength",
            field="range",
            id="strength",
        )
        width: int = Field(512, min=2, max=15, title="Width", disabled=True, hide=True, id="width")
        height: int = Field(512, min=2, max=15, title="Height", disabled=True, hide=True, id="height")

    def __init__(self, args: Args, device: torch.device, torch_dtype: torch.dtype):
        params = self.InputParams()
        
        # Optimized settings for SD-Turbo
        self.stream = StreamDiffusionWrapper(
            model_id_or_path=base_model,
            use_tiny_vae=args.taesd,
            device=device,
            dtype=torch_dtype,
            t_index_list=[1],  # Single step for speed
            frame_buffer_size=1,
            width=params.width,
            height=params.height,
            use_lcm_lora=False,
            output_type="pil",
            warmup=2,  # Reduced warmup
            vae_id=None,
            acceleration=args.acceleration,
            mode="img2img",
            use_denoising_batch=True,
            cfg_type="none",  # Fastest CFG type
            use_safety_checker=False,
            enable_similar_image_filter=True,  # Enable similarity filter
            similar_image_filter_threshold=0.95,  # High threshold
            similar_image_filter_max_skip_frame=3,  # Skip up to 3 frames
            engine_dir=args.engine_dir,
        )
        
        self.last_prompt = default_prompt
        self.stream.prepare(
            prompt=default_prompt,
            negative_prompt="",
            num_inference_steps=1,  # SD-Turbo optimized for 1 step
            guidance_scale=1.0,  # Low guidance for speed
        )

    def predict(self, params: "Pipeline.InputParams") -> Image.Image:
        # Update prompt if changed
        if params.prompt != self.last_prompt:
            self.stream.prepare(
                prompt=params.prompt,
                negative_prompt="",
                num_inference_steps=1,
                guidance_scale=1.0,
            )
            self.last_prompt = params.prompt
        
        image_tensor = self.stream.preprocess_image(params.image)
        output_image = self.stream(image=image_tensor, prompt=params.prompt)
        return output_image
