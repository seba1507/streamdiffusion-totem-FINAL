#!/usr/bin/env python3
"""
LCM-Optimized Server FIXED - No torch.compile conflicts
Target: 15-20 FPS on L40S
"""
import os
import sys
import time
import asyncio
import numpy as np
from PIL import Image
import torch
import gc
import logging

# CUDA optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress torch.compile errors
torch._dynamo.config.suppress_errors = True

if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler, AutoencoderTiny
import base64
import io
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class LCMProcessor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.pipe = None
        self.frame_count = 0
        self.last_time = time.time()
        
        self.init_pipeline()
        
    def init_pipeline(self):
        """Initialize LCM pipeline without torch.compile"""
        logger.info("🚀 Initializing LCM Pipeline...")
        
        # Use SD 1.5 for LCM
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Setup LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load LCM-LoRA
        try:
            logger.info("Loading LCM-LoRA...")
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.pipe.fuse_lora()
            logger.info("✅ LCM-LoRA loaded and fused!")
        except Exception as e:
            logger.error(f"❌ LCM-LoRA failed: {e}")
            logger.info("Installing peft...")
            os.system("pip install -q peft==0.7.1")
            
            # Retry
            try:
                self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                self.pipe.fuse_lora()
                logger.info("✅ LCM-LoRA loaded after installing peft!")
            except:
                logger.error("❌ LCM-LoRA failed completely. Running without it.")
        
        # TinyVAE for speed
        try:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype
            ).to(self.device)
            logger.info("✅ TinyVAE loaded")
        except:
            logger.warning("⚠️ TinyVAE failed")
        
        # Optimizations
        self.pipe.set_progress_bar_config(disable=True)
        
        # Memory efficient attention (choose one)
        try:
            # Option 1: XFormers
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ XFormers enabled")
        except:
            # Option 2: Slicing
            self.pipe.enable_attention_slicing(1)
            logger.info("✅ Attention slicing enabled")
        
        # VAE optimizations
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # NO torch.compile to avoid conflicts
        logger.info("⚠️ Skipping torch.compile to avoid conflicts")
        
        # Warmup
        self._warmup()
        
    def _warmup(self):
        """Warmup the pipeline"""
        logger.info("🔥 Warming up...")
        dummy = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        for i in range(2):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        prompt="test",
                        image=dummy,
                        num_inference_steps=4,
                        strength=0.5,
                        guidance_scale=1.0
                    ).images[0]
            logger.info(f"Warmup {i+1}/2")
        
        torch.cuda.empty_cache()
        logger.info("✅ Ready!")
        
    def process_frame(self, image, prompt="photo", strength=0.4):
        """Process frame with LCM"""
        start = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=4,  # LCM optimal
                    strength=strength,
                    guidance_scale=1.0      # LCM optimal
                ).images[0]
        
        # FPS tracking
        elapsed = time.time() - start
        self.frame_count += 1
        
        if time.time() - self.last_time >= 1.0:
            fps = self.frame_count / (time.time() - self.last_time)
            logger.info(f"📊 FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_time = time.time()
            
        return result, elapsed * 1000

# Global processor
processor = None

# Simple HTML
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>LCM Stream - Working</title>
    <style>
        body { background: #000; color: #0f0; font-family: monospace; margin: 0; padding: 20px; }
        video, canvas { width: 512px; height: 512px; border: 2px solid #0f0; margin: 10px; }
        button { background: #0f0; color: #000; border: none; padding: 15px 30px; cursor: pointer; margin: 5px; font-size: 16px; }
        button:disabled { background: #555; color: #999; }
        .stats { position: fixed; top: 10px; right: 10px; background: rgba(0,0,0,0.9); padding: 20px; border: 1px solid #0f0; }
        .fps { font-size: 24px; font-weight: bold; }
        input[type="range"] { width: 200px; }
    </style>
</head>
<body>
    <h1>LCM StreamDiffusion - L40S</h1>
    <div>
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    <div>
        <button onclick="start()">▶ Start</button>
        <button onclick="stop()">⏹ Stop</button>
        <label>Strength: <input type="range" id="strength" min="0.2" max="0.8" step="0.05" value="0.4"> 
        <span id="strengthVal">0.4</span></label>
    </div>
    <div class="stats">
        <div class="fps">FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
        <div>Frames: <span id="frames">0</span></div>
    </div>
    
    <script>
        let ws, video, canvas, ctx, streaming = false;
        let frameCount = 0, totalFrames = 0, lastTime = Date.now();
        
        document.getElementById('strength').oninput = function() {
            document.getElementById('strengthVal').textContent = this.value;
        };
        
        async function start() {
            video = document.getElementById('video');
            canvas = document.getElementById('output');
            ctx = canvas.getContext('2d');
            canvas.width = canvas.height = 512;
            
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            video.srcObject = stream;
            
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => { streaming = true; sendFrame(); };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0, 512, 512);
                img.src = data.image;
                
                frameCount++;
                totalFrames++;
                document.getElementById('frames').textContent = totalFrames;
                document.getElementById('latency').textContent = Math.round(data.time);
                
                const now = Date.now();
                if (now - lastTime > 1000) {
                    document.getElementById('fps').textContent = frameCount;
                    frameCount = 0;
                    lastTime = now;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            const temp = document.createElement('canvas');
            temp.width = temp.height = 512;
            const tctx = temp.getContext('2d');
            
            // Center crop
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            tctx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: temp.toDataURL('image/jpeg', 0.8),
                strength: parseFloat(document.getElementById('strength').value)
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
    logger.info("Client connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Decode image
            img_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            if image.size != (512, 512):
                image = image.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Process
            result, elapsed = processor.process_frame(
                image,
                prompt="high quality professional photo",
                strength=data.get('strength', 0.4)
            )
            
            # Encode
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}',
                'time': elapsed
            })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")

@app.on_event("startup")
async def startup():
    global processor
    processor = LCMProcessor()

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 LCM STREAMDIFFUSION - WORKING VERSION")
    print("="*80)
    print("✅ Target: 15-20 FPS")
    print("✅ Fixed: torch.compile disabled to avoid conflicts")
    print("✅ Model: SD 1.5 + LCM-LoRA (4 steps)")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
