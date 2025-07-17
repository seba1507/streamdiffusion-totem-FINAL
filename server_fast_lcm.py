#!/usr/bin/env python3
"""
Fast LCM StreamDiffusion - Optimized for L40S (15-20 FPS)
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

# Configure CUDA for L40S
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# Verify CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# Enable optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(0.8)

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

class FastLCMProcessor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.pipe = None
        
        # Performance tracking
        self.frame_count = 0
        self.last_time = time.time()
        
        self.init_model()
        
    def init_model(self):
        """Initialize LCM-optimized pipeline"""
        logger.info("🚀 Initializing Fast LCM Pipeline...")
        
        # Load base model (SD 1.5 is faster than SD-Turbo for LCM)
        model_id = "runwayml/stable-diffusion-v1-5"
        
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)
        
        # CRITICAL: Configure LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load LCM-LoRA weights
        try:
            logger.info("Loading LCM-LoRA...")
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.pipe.fuse_lora()  # Fuse for better performance
            logger.info("✅ LCM-LoRA loaded and fused")
        except Exception as e:
            logger.error(f"❌ LCM-LoRA failed: {e}")
            sys.exit(1)
        
        # Use Tiny AutoEncoder
        try:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype
            ).to(self.device)
            logger.info("✅ TinyVAE loaded")
        except:
            logger.warning("⚠️ TinyVAE failed, using default")
        
        # Optimizations
        self.pipe.set_progress_bar_config(disable=True)
        
        # Enable XFormers if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ XFormers enabled")
        except:
            logger.warning("⚠️ XFormers not available, using slicing")
            self.pipe.enable_attention_slicing(1)
        
        # VAE optimizations
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Compile UNet if available
        if hasattr(torch, 'compile'):
            try:
                self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
                logger.info("✅ UNet compiled")
            except:
                pass
        
        # Warmup
        self._warmup()
        
    def _warmup(self):
        """Warmup the pipeline"""
        logger.info("🔥 Warming up...")
        dummy = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        for _ in range(2):
            with torch.no_grad():
                self.pipe(
                    prompt="warmup",
                    image=dummy,
                    num_inference_steps=4,
                    strength=0.5,
                    guidance_scale=1.0
                ).images[0]
        
        torch.cuda.empty_cache()
        logger.info("✅ Ready!")
        
    def process_frame(self, image, prompt="high quality", strength=0.4):
        """Process a single frame with LCM"""
        # LCM optimal parameters
        guidance_scale = 1.0  # LCM works best with low CFG
        num_steps = 4  # LCM is designed for 4 steps
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=num_steps,
                    strength=strength,
                    guidance_scale=guidance_scale
                ).images[0]
        
        # Track FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            fps = self.frame_count / (current_time - self.last_time)
            logger.info(f"FPS: {fps:.1f}")
            self.frame_count = 0
            self.last_time = current_time
            
        return result

# Global processor
processor = None

# Simple HTML interface
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fast LCM StreamDiffusion</title>
    <style>
        body {
            background: #1a1a1a;
            color: #fff;
            font-family: Arial, sans-serif;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            text-align: center;
        }
        video, canvas {
            width: 512px;
            height: 512px;
            margin: 10px;
            border: 2px solid #00ff88;
            background: #000;
        }
        .controls {
            margin-top: 20px;
        }
        button {
            padding: 10px 30px;
            font-size: 16px;
            background: #00ff88;
            border: none;
            color: #000;
            cursor: pointer;
            margin: 5px;
        }
        button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.8);
            padding: 20px;
            border: 1px solid #00ff88;
        }
        input[type="range"] {
            width: 200px;
        }
        .presets {
            margin: 10px 0;
        }
        .preset-btn {
            background: #444;
            padding: 8px 20px;
            margin: 2px;
        }
        .preset-btn.active {
            background: #00ff88;
            color: #000;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="color: #00ff88;">Fast LCM StreamDiffusion</h1>
        <div>
            <video id="video" autoplay muted playsinline></video>
            <canvas id="output"></canvas>
        </div>
        
        <div class="controls">
            <div class="presets">
                <button class="preset-btn active" onclick="setPreset('fast')">Fast Mode</button>
                <button class="preset-btn" onclick="setPreset('quality')">Quality</button>
                <button class="preset-btn" onclick="setPreset('artistic')">Artistic</button>
            </div>
            
            <div style="margin: 10px;">
                <label>Strength: <span id="strengthValue">0.4</span></label><br>
                <input type="range" id="strength" min="0.2" max="0.8" step="0.05" value="0.4" 
                       oninput="document.getElementById('strengthValue').textContent = this.value">
            </div>
            
            <div style="margin: 10px;">
                <input type="text" id="prompt" placeholder="Enter prompt..." 
                       value="high quality professional photo" style="width: 300px; padding: 5px;">
            </div>
            
            <button id="startBtn" onclick="start()">Start</button>
            <button id="stopBtn" onclick="stop()" disabled>Stop</button>
        </div>
    </div>
    
    <div class="stats">
        <h3 style="margin-top: 0;">Performance</h3>
        <div>FPS: <span id="fps" style="color: #00ff88;">0</span></div>
        <div>Latency: <span id="latency" style="color: #00ff88;">0</span>ms</div>
        <div>Frames: <span id="frames" style="color: #00ff88;">0</span></div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video, canvas, ctx;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        const presets = {
            fast: { strength: 0.4, prompt: "high quality professional photo" },
            quality: { strength: 0.6, prompt: "highly detailed professional photography" },
            artistic: { strength: 0.7, prompt: "artistic creative style" }
        };
        
        let currentPreset = 'fast';
        
        function setPreset(preset) {
            currentPreset = preset;
            const settings = presets[preset];
            document.getElementById('strength').value = settings.strength;
            document.getElementById('strengthValue').textContent = settings.strength;
            document.getElementById('prompt').value = settings.prompt;
            
            document.querySelectorAll('.preset-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        async function start() {
            video = document.getElementById('video');
            canvas = document.getElementById('output');
            ctx = canvas.getContext('2d');
            canvas.width = 512;
            canvas.height = 512;
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480 } 
            });
            video.srcObject = stream;
            
            await new Promise(resolve => video.onloadedmetadata = resolve);
            
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            ws.onopen = () => {
                streaming = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                sendFrame();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => ctx.drawImage(img, 0, 0, 512, 512);
                    img.src = data.image;
                    updateStats(data);
                }
            };
            
            ws.onerror = ws.onclose = () => stop();
        }
        
        function sendFrame() {
            if (!streaming || ws.readyState !== WebSocket.OPEN) return;
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Center crop
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.8);
            
            ws.send(JSON.stringify({
                image: imageData,
                prompt: document.getElementById('prompt').value,
                strength: parseFloat(document.getElementById('strength').value),
                timestamp: Date.now()
            }));
            
            requestAnimationFrame(sendFrame);
        }
        
        function updateStats(data) {
            frameCount++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 30) latencies.shift();
            }
            
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
                
                if (latencies.length > 0) {
                    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avgLatency);
                }
            }
            
            document.getElementById('frames').textContent = 
                parseInt(document.getElementById('frames').textContent) + 1;
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
                
            # Decode image
            img_data = base64.b64decode(data['image'].split(',')[1])
            image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Process
            start_time = time.time()
            
            result = processor.process_frame(
                image,
                prompt=data.get('prompt', 'high quality'),
                strength=data.get('strength', 0.4)
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Encode result
            buffered = io.BytesIO()
            result.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Send back
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}',
                'processing_time': processing_time,
                'timestamp': data.get('timestamp', 0)
            })
            
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")

@app.on_event("startup")
async def startup():
    global processor
    processor = FastLCMProcessor()

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 FAST LCM STREAMDIFFUSION")
    print("="*80)
    print("✅ Target: 15-20 FPS")
    print("✅ Model: SD 1.5 + LCM-LoRA")
    print("✅ Steps: 4 (optimized)")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
