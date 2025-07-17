#!/usr/bin/env python3
"""
LCM-Optimized Server for L40S - Target: 15-20 FPS
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
from collections import deque

# CUDA optimizations for L40S
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

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

class LCMFastProcessor:
    def __init__(self):
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.pipe = None
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        self.init_pipeline()
        
    def init_pipeline(self):
        """Initialize LCM-optimized pipeline"""
        logger.info("🚀 Initializing LCM-Optimized Pipeline...")
        
        # Use SD 1.5 (not SD-Turbo) for LCM
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Load base pipeline
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)
        
        # CRITICAL: Setup LCM Scheduler BEFORE loading LoRA
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            prediction_type="epsilon"
        )
        
        # Load and fuse LCM-LoRA
        try:
            logger.info("Loading LCM-LoRA...")
            # Install peft if needed
            try:
                import peft
            except ImportError:
                logger.info("Installing peft...")
                os.system("pip install peft==0.7.1")
                import peft
            
            # Load LCM-LoRA weights
            self.pipe.load_lora_weights(
                "latent-consistency/lcm-lora-sdv1-5",
                weight_name="pytorch_lora_weights.safetensors"
            )
            
            # Fuse LoRA weights for speed
            self.pipe.fuse_lora(lora_scale=1.0)
            logger.info("✅ LCM-LoRA loaded and fused successfully!")
            
        except Exception as e:
            logger.error(f"❌ LCM-LoRA loading failed: {e}")
            logger.info("Attempting alternative loading method...")
            
            # Alternative: Direct LoRA loading
            try:
                from diffusers import StableDiffusionPipeline
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=self.dtype,
                    safety_checker=None
                ).to(self.device)
                
                self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
                self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
                self.pipe.fuse_lora()
                logger.info("✅ Alternative loading successful!")
            except Exception as e2:
                logger.error(f"Alternative loading also failed: {e2}")
        
        # Load TinyVAE for faster decoding
        try:
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype
            ).to(self.device)
            logger.info("✅ TinyVAE loaded for faster decoding")
        except:
            logger.warning("⚠️ TinyVAE failed, using default VAE")
        
        # Pipeline optimizations
        self.pipe.set_progress_bar_config(disable=True)
        
        # Enable memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ XFormers enabled")
        except:
            self.pipe.enable_attention_slicing(1)
            logger.info("✅ Attention slicing enabled")
        
        # VAE optimizations
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # Compile if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.pipe.unet = torch.compile(
                    self.pipe.unet, 
                    mode="reduce-overhead",
                    fullgraph=False
                )
                logger.info("✅ UNet compiled with torch.compile")
            except:
                logger.info("⚠️ torch.compile not available")
        
        # Warmup
        self._warmup()
        
    def _warmup(self):
        """Warmup the pipeline"""
        logger.info("🔥 Warming up pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        # Warmup with LCM settings
        for i in range(3):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        prompt="a photo",
                        image=dummy_image,
                        num_inference_steps=4,  # LCM uses 4 steps
                        strength=0.5,
                        guidance_scale=1.0,  # LCM uses low CFG
                        generator=torch.Generator(device=self.device).manual_seed(42)
                    ).images[0]
            logger.info(f"Warmup {i+1}/3 complete")
        
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Pipeline ready for real-time processing!")
        
    def process_frame(self, image, prompt="high quality photo", strength=0.4):
        """Process frame with LCM optimization"""
        # LCM optimal parameters
        num_inference_steps = 4  # LCM is designed for 4 steps
        guidance_scale = 1.0     # Low CFG for LCM (1.0-2.0 range)
        
        # Time tracking
        start_time = time.time()
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = self.pipe(
                    prompt=prompt,
                    image=image,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).images[0]
        
        # Calculate FPS
        process_time = time.time() - start_time
        fps = 1.0 / process_time
        self.fps_history.append(fps)
        
        # Log performance every 30 frames
        if len(self.fps_history) == 30:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            logger.info(f"📊 Average FPS: {avg_fps:.1f}")
            
        return result, process_time * 1000

# Global processor
processor = None

# Optimized HTML interface
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>LCM StreamDiffusion - L40S Optimized</title>
    <style>
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            overflow: hidden;
        }
        
        .main-container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 40px;
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            border: 2px solid #00ff88;
            background: #111;
            border-radius: 16px;
            box-shadow: 0 0 40px rgba(0, 255, 136, 0.3);
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(20, 20, 20, 0.95);
            padding: 25px 40px;
            border-radius: 20px;
            border: 1px solid #333;
            backdrop-filter: blur(10px);
            display: flex;
            gap: 30px;
            align-items: center;
        }
        
        button {
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            background: #00ff88;
            color: #000;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4);
        }
        
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(20, 20, 20, 0.95);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid #333;
            font-family: 'SF Mono', monospace;
            font-size: 14px;
            min-width: 250px;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid #222;
        }
        
        .stat-value {
            color: #00ff88;
            font-weight: 600;
        }
        
        .slider-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        
        input[type="range"] {
            width: 150px;
            height: 6px;
            background: #333;
            outline: none;
            border-radius: 3px;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 18px;
            height: 18px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
        }
        
        .title {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 24px;
            font-weight: 700;
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        
        .fps-indicator {
            font-size: 32px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 15px;
        }
        
        .fps-good { color: #00ff88; }
        .fps-ok { color: #ffaa00; }
        .fps-bad { color: #ff4444; }
    </style>
</head>
<body>
    <div class="title">LCM StreamDiffusion Pro</div>
    
    <div class="main-container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    
    <div class="stats">
        <div class="fps-indicator" id="fpsIndicator">-- FPS</div>
        <div class="stat-row">
            <span>Latency:</span>
            <span class="stat-value" id="latency">--</span>
        </div>
        <div class="stat-row">
            <span>Frames:</span>
            <span class="stat-value" id="frames">0</span>
        </div>
        <div class="stat-row">
            <span>Target:</span>
            <span class="stat-value">15-20 FPS</span>
        </div>
        <div class="stat-row">
            <span>Quality:</span>
            <span class="stat-value">LCM 4-step</span>
        </div>
    </div>
    
    <div class="controls">
        <button id="startBtn" onclick="start()">🚀 Start</button>
        <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
        
        <div class="slider-container">
            <label>Strength:</label>
            <input type="range" id="strength" min="0.2" max="0.6" step="0.05" value="0.4">
            <span id="strengthValue">0.4</span>
        </div>
        
        <input type="text" id="prompt" placeholder="Prompt..." value="high quality professional photo" 
               style="padding: 10px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #fff;">
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video, canvas, ctx;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        // Update strength value display
        document.getElementById('strength').oninput = function() {
            document.getElementById('strengthValue').textContent = this.value;
        };
        
        async function start() {
            video = document.getElementById('video');
            canvas = document.getElementById('output');
            ctx = canvas.getContext('2d', { alpha: false });
            canvas.width = 512;
            canvas.height = 512;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    }
                });
                video.srcObject = stream;
                
                await new Promise(resolve => video.onloadedmetadata = resolve);
                
                // Connect WebSocket
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Connected to LCM server');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0, 512, 512);
                        };
                        img.src = data.image;
                        updateStats(data);
                    }
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    stop();
                };
                
                ws.onclose = () => {
                    console.log('Disconnected');
                    stop();
                };
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error accessing camera: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                requestAnimationFrame(sendFrame);
                return;
            }
            
            // Capture and resize frame
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Center crop
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            // Send frame
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.85);
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
            document.getElementById('frames').textContent = frameCount;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 30) latencies.shift();
                
                const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                document.getElementById('latency').textContent = Math.round(avgLatency) + 'ms';
            }
            
            // Update FPS
            const now = Date.now();
            if (now - lastTime >= 1000) {
                const fps = frameCount / ((now - lastTime) / 1000);
                const fpsElement = document.getElementById('fpsIndicator');
                fpsElement.textContent = fps.toFixed(1) + ' FPS';
                
                // Color code FPS
                if (fps >= 15) {
                    fpsElement.className = 'fps-indicator fps-good';
                } else if (fps >= 10) {
                    fpsElement.className = 'fps-indicator fps-ok';
                } else {
                    fpsElement.className = 'fps-indicator fps-bad';
                }
                
                frameCount = 0;
                lastTime = now;
            }
        }
        
        function stop() {
            streaming = false;
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        // Handle page unload
        window.addEventListener('beforeunload', stop);
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
    logger.info("✅ Client connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
            
            try:
                # Decode image
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Process frame
                result_image, processing_time = processor.process_frame(
                    input_image,
                    prompt=data.get('prompt', 'high quality photo'),
                    strength=data.get('strength', 0.4)
                )
                
                # Encode result
                buffered = io.BytesIO()
                result_image.save(buffered, format="JPEG", quality=85, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send result
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'processing_time': processing_time,
                    'timestamp': data.get('timestamp', 0)
                })
                
            except Exception as e:
                logger.error(f"Frame processing error: {e}")
                continue
                
    except WebSocketDisconnect:
        logger.info("❌ Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.on_event("startup")
async def startup_event():
    global processor
    processor = LCMFastProcessor()
    logger.info("🚀 LCM StreamDiffusion server started")

@app.on_event("shutdown")
async def shutdown_event():
    if processor:
        torch.cuda.empty_cache()
        gc.collect()
    logger.info("Server stopped")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 LCM STREAMDIFFUSION - L40S OPTIMIZED")
    print("="*80)
    print("✅ Target: 15-20 FPS")
    print("✅ Model: SD 1.5 + LCM-LoRA (4 steps)")
    print("✅ Optimizations: TinyVAE, XFormers, Mixed Precision")
    print("✅ Resolution: 512x512")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=False
    )
