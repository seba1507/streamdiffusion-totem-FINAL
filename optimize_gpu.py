#!/usr/bin/env python3
"""
Script para optimizar GPU y encontrar la mejor configuración
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import time
from diffusers import AutoPipelineForImage2Image, LCMScheduler
from PIL import Image

print("="*60)
print("🔧 GPU OPTIMIZATION TEST")
print("="*60)

# Check GPU
print(f"\n📊 GPU Info:")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print(f"Compute Capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
print("\n✅ Set memory fraction to 90%")

# Enable TF32 for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:  # Ampere or newer
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✅ TF32 enabled for Ampere GPU")

# Test different configurations
print("\n🧪 Testing configurations...")

# Load model
model_id = "SimianLuo/LCM_Dreamshaper_v7"
pipe = AutoPipelineForImage2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,
    use_safetensors=True
).to("cuda")

pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.set_progress_bar_config(disable=True)

# Try XFormers
try:
    pipe.enable_xformers_memory_efficient_attention()
    xformers = True
    print("✅ XFormers enabled")
except:
    xformers = False
    print("❌ XFormers not available")

# Test configurations
configs = [
    {"res": 256, "steps": 1, "strength": 0.3},
    {"res": 384, "steps": 1, "strength": 0.3},
    {"res": 512, "steps": 1, "strength": 0.3},
    {"res": 384, "steps": 1, "strength": 0.5},
    {"res": 384, "steps": 2, "strength": 0.3},
]

print("\n📊 Benchmark Results:")
print("-" * 60)
print(f"{'Config':<30} {'Time (ms)':<15} {'FPS':<10}")
print("-" * 60)

dummy_prompt = "cyberpunk portrait"

for config in configs:
    # Create test image
    test_img = Image.new('RGB', (config["res"], config["res"]), color=(128, 128, 128))
    
    # Warm up
    with torch.no_grad():
        _ = pipe(
            dummy_prompt,
            image=test_img,
            num_inference_steps=config["steps"],
            strength=config["strength"],
            guidance_scale=1.0
        ).images[0]
    
    # Benchmark
    times = []
    for _ in range(5):
        start = time.time()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                _ = pipe(
                    dummy_prompt,
                    image=test_img,
                    num_inference_steps=config["steps"],
                    strength=config["strength"],
                    guidance_scale=1.0
                ).images[0]
        torch.cuda.synchronize()
        times.append((time.time() - start) * 1000)
    
    avg_time = sum(times) / len(times)
    fps = 1000 / avg_time
    
    config_str = f"{config['res']}px, {config['steps']}step, str={config['strength']}"
    print(f"{config_str:<30} {avg_time:>10.1f} ms {fps:>8.1f}")

print("-" * 60)

# Memory optimization tips
print("\n💡 Recommendations:")
if avg_time > 100:
    print("- Use 384px or 256px resolution for better FPS")
    print("- Keep strength at 0.3 or lower")
    print("- Use only 1 inference step")
if not xformers:
    print("- Install xformers for 20-30% speedup:")
    print("  pip install xformers==0.0.23")

# Clear cache
torch.cuda.empty_cache()
print("\n✅ GPU cache cleared")
print("="*60)
