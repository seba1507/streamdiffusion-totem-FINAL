#!/usr/bin/env python3
"""
Simple Working Server - No conflicts, guaranteed to work
Target: 10-15 FPS minimum
"""
import os
import sys
import time
import torch
import numpy as np
from PIL import Image

# Basic CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
import base64
import io

app = FastAPI()

class SimpleProcessor:
    def __init__(self):
        self.device = "cuda"
        self.pipe = None
        self.init_pipeline()
        
    def init_pipeline(self):
        print("🚀 Loading pipeline...")
        
        # Use SD 1.5 - most compatible
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Fast scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # CRITICAL: Use attention slicing instead of XFormers to avoid conflicts
        self.pipe.enable_attention_slicing(1)
        print("✅ Using attention slicing (no XFormers conflicts)")
        
        # VAE optimizations (no TinyVAE to avoid conflicts)
        self.pipe.vae.enable_slicing()
        print("✅ VAE slicing enabled")
        
        # Disable progress bar
        self.pipe.set_progress_bar_config(disable=True)
        
        # Warmup
        print("🔥 Warming up...")
        dummy = Image.new('RGB', (512, 512), color=(128, 128, 128))
        with torch.no_grad():
            self.pipe(
                prompt="test",
                image=dummy,
                num_inference_steps=10,
                strength=0.5,
                guidance_scale=7.5
            )
        
        torch.cuda.empty_cache()
        print("✅ Ready!")
        
    def process(self, image, prompt="photo", strength=0.5, steps=10):
        with torch.no_grad():
            # Ensure consistent dtype
            with torch.cuda.amp.autocast(enabled=False):  # Disable autocast to avoid dtype issues
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=7.5
                ).images[0]
        return result

processor = SimpleProcessor()
frame_count = 0
last_time = time.time()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple StreamDiffusion</title>
    <style>
        body { background: #1a1a1a; color: #fff; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; align-items: center; justify-content: center; }
        video, canvas { width: 512px; height: 512px; border: 2px solid #00ff88; }
        button { background: #00ff88; color: #000; border: none; padding: 15px 30px; font-size: 16px; cursor: pointer; margin: 10px; }
        .stats { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8); padding: 20px; border: 1px solid #00ff88; }
        .controls { text-align: center; margin-top: 20px; }
        input[type="range"] { width: 200px; margin: 0 10px; }
    </style>
</head>
<body>
    <h1 style="text-align: center;">Simple StreamDiffusion - L40S</h1>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    <div class="controls">
        <button onclick="start()">▶ Start</button>
        <button onclick="stop()">⏹ Stop</button>
        <br><br>
        <label>Strength: <input type="range" id="strength" min="0.3" max="0.8" step="0.05" value="0.5">
        <span id="strengthVal">0.5</span></label>
        <br>
        <label>Steps: <input type="range" id="steps" min="5" max="20" step="1" value="10">
        <span id="stepsVal">10</span></label>
    </div>
    <div class="stats">
        <div style="font-size: 24px; font-weight: bold;">FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
        <div>Model: SD 1.5</div>
        <div>No conflicts mode</div>
    </div>
    
    <script>
        let ws, video, canvas, ctx, streaming = false;
        let frames = 0, lastTime = Date.now();
        
        document.getElementById('strength').oninput = function() {
            document.getElementById('strengthVal').textContent = this.value;
        };
        document.getElementById('steps').oninput = function() {
            document.getElementById('stepsVal').textContent = this.value;
        };
        
        async function start() {
            video = document.getElementById('video');
            canvas = document.getElementById('output');
            ctx = canvas.getContext('2d');
            canvas.width = canvas.height = 512;
            
            video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
            
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => { streaming = true; sendFrame(); };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0, 512, 512);
                img.src = data.image;
                
                document.getElementById('latency').textContent = Math.round(data.time);
                
                frames++;
                const now = Date.now();
                if (now - lastTime > 1000) {
                    document.getElementById('fps').textContent = frames;
                    frames = 0;
                    lastTime = now;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            const temp = document.createElement('canvas');
            temp.width = temp.height = 512;
            temp.getContext('2d').drawImage(video, 0, 0, 512, 512);
            ws.send(JSON.stringify({
                image: temp.toDataURL('image/jpeg', 0.8),
                strength: parseFloat(document.getElementById('strength').value),
                steps: parseInt(document.getElementById('steps').value)
            }));
            requestAnimationFrame(sendFrame);
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video && video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global frame_count, last_time
    
    while True:
        data = await websocket.receive_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        start = time.time()
        result = processor.process(
            image,
            prompt="high quality photo",
            strength=data.get('strength', 0.5),
            steps=data.get('steps', 10)
        )
        elapsed = (time.time() - start) * 1000
        
        # FPS tracking
        frame_count += 1
        if time.time() - last_time >= 5:
            fps = frame_count / (time.time() - last_time)
            print(f"📊 Average FPS: {fps:.1f}")
            frame_count = 0
            last_time = time.time()
        
        buffered = io.BytesIO()
        result.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        await websocket.send_json({'image': f'data:image/jpeg;base64,{img_str}', 'time': elapsed})

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 SIMPLE STREAMDIFFUSION - NO CONFLICTS")
    print("="*80)
    print("✅ Model: SD 1.5")
    print("✅ No XFormers (using attention slicing)")
    print("✅ No TinyVAE (using default VAE)")
    print("✅ No autocast (avoiding dtype conflicts)")
    print("🎯 Target: 10-15 FPS minimum")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
