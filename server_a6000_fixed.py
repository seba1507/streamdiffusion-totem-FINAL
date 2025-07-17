#!/usr/bin/env python3
import os
import sys
import time
import threading
import queue
import numpy as np
from PIL import Image

# Configurar CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch

# Optimizaciones para A6000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import AutoPipelineForImage2Image, LCMScheduler
import base64
import io
import json

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class A6000FastDiffusion:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.resolution = 384  # Mejor para A6000
        
        # Threading simple
        self.current_task = None
        self.last_result = None
        self.processing = False
        
        # Counters
        self.total_frames = 0
        self.processed_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("\n🚀 Inicializando A6000 Fast Diffusion...")
        
        # LCM Dreamshaper - probado y funcional
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        
        print(f"📥 Cargando {model_id}...")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)
        
        # LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers - crítico para velocidad
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers habilitado")
        except Exception as e:
            print(f"❌ XFormers error: {e}")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✅ VAE optimizado")
        except:
            pass
        
        # Warmup simple
        print("🔥 Calentando pipeline...")
        try:
            dummy = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))
            with torch.no_grad():
                _ = self.pipe(
                    "test",
                    image=dummy,
                    num_inference_steps=1,
                    strength=0.3,
                    guidance_scale=1.0
                ).images[0]
            torch.cuda.empty_cache()
            print("✅ Pipeline listo!")
        except Exception as e:
            print(f"⚠️  Warmup error (no crítico): {e}")
        
        # Start processing
        self.processing = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("✅ Processing thread iniciado\n")
    
    def _process_loop(self):
        """Loop de procesamiento simple y robusto"""
        while self.processing:
            if self.current_task is None:
                time.sleep(0.001)
                continue
            
            try:
                start = time.time()
                
                # Get task data
                task = self.current_task
                image = task['image']
                
                # Resize if needed
                if image.size != (self.resolution, self.resolution):
                    image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                
                # Generate
                with torch.no_grad():
                    result = self.pipe(
                        prompt=task.get('prompt', '') or '',
                        image=image,
                        num_inference_steps=1,
                        strength=min(0.4, task.get('strength', 0.3)),
                        guidance_scale=1.0
                    ).images[0]
                
                # Scale to 512
                if result.size != (512, 512):
                    result = result.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Save result
                elapsed = (time.time() - start) * 1000
                self.processed_frames += 1
                
                self.last_result = {
                    'image': result,
                    'time': elapsed,
                    'stats': {
                        'total': self.total_frames,
                        'processed': self.processed_frames,
                        'skip_rate': round((1 - self.processed_frames / max(1, self.total_frames)) * 100, 1)
                    }
                }
                
                print(f"✓ Frame: {elapsed:.0f}ms")
                
            except Exception as e:
                print(f"❌ Error: {type(e).__name__}: {e}")
            
            self.current_task = None
            
            # Clean cache every 30 frames
            if self.processed_frames % 30 == 0:
                torch.cuda.empty_cache()
    
    def add_frame(self, image, prompt, strength):
        self.total_frames += 1
        
        # Skip if busy
        if self.current_task is not None:
            return
        
        self.current_task = {
            'image': image,
            'prompt': prompt,
            'strength': strength
        }
    
    def get_result(self):
        return self.last_result

# Global instance
processor = A6000FastDiffusion()

# Simple HTML
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>A6000 Fast Diffusion</title>
    <style>
        body {
            background: #000;
            color: #0f0;
            font-family: monospace;
            margin: 0;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 30px;
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            border: 2px solid #0f0;
            background: #111;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border: 2px solid #0f0;
            border-radius: 10px;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        button {
            padding: 10px 20px;
            background: #0f0;
            color: #000;
            border: none;
            font-weight: bold;
            cursor: pointer;
            text-transform: uppercase;
        }
        
        button:disabled {
            background: #333;
            color: #666;
        }
        
        input[type="range"] {
            width: 150px;
        }
        
        textarea {
            flex: 1;
            padding: 10px;
            background: #111;
            color: #0f0;
            border: 1px solid #0f0;
            font-family: monospace;
            resize: none;
            height: 40px;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border: 2px solid #0f0;
            font-size: 14px;
            min-width: 200px;
        }
        
        .totem-mode {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
            justify-content: center;
            align-items: center;
        }
        
        .totem-canvas {
            max-width: 90%;
            max-height: 90%;
        }
        
        .totem-exit {
            position: absolute;
            top: 20px;
            right: 20px;
            width: 50px;
            height: 50px;
            background: #f00;
            color: #fff;
            border: none;
            border-radius: 50%;
            font-size: 20px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    
    <div class="totem-mode" id="totemMode">
        <button class="totem-exit" onclick="exitTotem()">✕</button>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
    </div>
    
    <div class="stats">
        <div>FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
        <div>Frames: <span id="frames">0</span></div>
        <div>Skip: <span id="skip">0</span>%</div>
    </div>
    
    <div class="controls">
        <button id="startBtn" onclick="start()">START</button>
        <button id="stopBtn" onclick="stop()" disabled>STOP</button>
        <button onclick="enterTotem()">TOTEM</button>
        <textarea id="prompt" placeholder="Enter prompt...">cyberpunk portrait</textarea>
        <label>STR: <span id="strVal">0.3</span></label>
        <input type="range" id="strength" min="0.2" max="0.4" step="0.05" value="0.3" 
               oninput="document.getElementById('strVal').textContent=this.value">
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        function enterTotem() {
            document.getElementById('totemMode').style.display = 'flex';
            const tc = document.getElementById('totemCanvas');
            tc.width = 512;
            tc.height = 512;
        }
        
        function exitTotem() {
            document.getElementById('totemMode').style.display = 'none';
        }
        
        async function start() {
            try {
                const video = document.getElementById('video');
                const canvas = document.getElementById('output');
                const ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                video.srcObject = stream;
                
                await new Promise(r => video.onloadedmetadata = r);
                
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
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0);
                            
                            // Totem mode
                            if (document.getElementById('totemMode').style.display === 'flex') {
                                const tc = document.getElementById('totemCanvas').getContext('2d');
                                tc.drawImage(img, 0, 0);
                            }
                        };
                        img.src = data.image;
                        updateStats(data);
                    }
                };
                
                ws.onerror = ws.onclose = () => stop();
                
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== 1) return;
            
            const video = document.getElementById('video');
            if (!video.videoWidth) {
                setTimeout(sendFrame, 10);
                return;
            }
            
            const tc = document.createElement('canvas');
            tc.width = tc.height = 512;
            const ctx = tc.getContext('2d');
            
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            ctx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: tc.toDataURL('image/jpeg', 0.8),
                prompt: document.getElementById('prompt').value,
                strength: parseFloat(document.getElementById('strength').value)
            }));
            
            setTimeout(sendFrame, 40); // ~25 FPS input
        }
        
        function updateStats(data) {
            frameCount++;
            
            if (data.time) {
                latencies.push(data.time);
                if (latencies.length > 30) latencies.shift();
            }
            
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount;
                frameCount = 0;
                lastTime = now;
                
                if (latencies.length) {
                    const avg = latencies.reduce((a,b) => a+b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avg);
                }
            }
            
            if (data.stats) {
                document.getElementById('frames').textContent = data.stats.total;
                document.getElementById('skip').textContent = data.stats.skip_rate;
            }
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            
            const video = document.getElementById('video');
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
                video.srcObject = null;
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') exitTotem();
            if (e.key === 'F11') { e.preventDefault(); enterTotem(); }
        });
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
    print("Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                # Decode image
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
            except Exception as e:
                print(f"Error imagen: {e}")
                continue
            
            # Process
            processor.add_frame(
                image,
                data.get('prompt', ''),
                data.get('strength', 0.3)
            )
            
            # Get result
            result = processor.get_result()
            if result:
                # Encode
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'time': result['time'],
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("="*60)
    print("🚀 A6000 FAST DIFFUSION - SIMPLIFIED")
    print("="*60)
    print("📐 Resolution: 384x384 (optimal for A6000)")
    print("⚡ 1-step LCM inference")
    print("🎯 Target: 10+ FPS")
    print("🌐 http://0.0.0.0:8000")
    print("="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
