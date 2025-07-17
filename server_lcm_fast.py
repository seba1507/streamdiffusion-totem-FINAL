#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import numpy as np
from PIL import Image
import cv2

# Configurar CUDA antes de importar torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")

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

class FastLCMDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Processing optimizado
        self.processing = False
        self.last_result = None
        self.processing_lock = threading.Lock()
        
        # Metrics
        self.total_frames = 0
        self.processed_frames = 0
        self.skip_counter = 0
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando Fast LCM Diffusion...")
        
        # Usar Dreamshaper LCM (ya descargado)
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # LCM Scheduler optimizado
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Optimizaciones críticas
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ XFormers habilitado")
        except:
            print("⚠ XFormers no disponible")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado")
        except:
            pass
        
        # Pre-calentamiento
        print("Pre-calentando pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        "test",
                        image=dummy_image,
                        num_inference_steps=1,
                        strength=0.5,
                        guidance_scale=1.0
                    ).images[0]
            print("✓ Pre-calentamiento exitoso")
        except Exception as e:
            print(f"⚠ Pre-calentamiento falló: {e}")
        
        print("✓ Fast LCM Diffusion listo!")
        self.start_processing_thread()
    
    def start_processing_thread(self):
        self.processing = True
        self.current_task = None
        processing_thread = threading.Thread(target=self._processing_loop_async, daemon=True)
        processing_thread.start()
    
    def _processing_loop_async(self):
        """Loop asíncrono optimizado"""
        while self.processing:
            if self.current_task:
                try:
                    start_time = time.time()
                    
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            result = self.pipe(
                                prompt=self.current_task['prompt'],
                                image=self.current_task['image'],
                                num_inference_steps=1,
                                strength=self.current_task['strength'],
                                guidance_scale=1.0
                            ).images[0]
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.processed_frames += 1
                    
                    with self.processing_lock:
                        self.last_result = {
                            'image': result,
                            'processing_time': processing_time,
                            'timestamp': self.current_task['timestamp'],
                            'stats': {
                                'total_frames': self.total_frames,
                                'processed_frames': self.processed_frames,
                                'skip_rate': (1 - self.processed_frames / max(1, self.total_frames)) * 100
                            }
                        }
                    
                    print(f"Frame: {processing_time:.1f}ms")
                    self.current_task = None
                    
                except Exception as e:
                    print(f"Error: {e}")
                    self.current_task = None
            else:
                time.sleep(0.001)  # 1ms sleep
    
    def process_frame(self, image, prompt, strength, timestamp):
        self.total_frames += 1
        
        # Skip si ya estamos procesando
        if self.current_task is not None:
            self.skip_counter += 1
            return
        
        self.current_task = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'timestamp': timestamp
        }
    
    def get_latest_result(self):
        with self.processing_lock:
            return self.last_result

# Instancia global
processor = FastLCMDiffusion()

# HTML optimizado
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Fast LCM StreamDiffusion</title>
    <style>
        body {
            background: #000;
            color: #fff;
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
            background: rgba(0, 0, 0, 0.9);
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #0f0;
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background: #0f0;
            color: #000;
            border: none;
            border-radius: 5px;
            font-weight: bold;
        }
        
        button:disabled {
            background: #333;
            color: #666;
        }
        
        input[type="range"] {
            width: 200px;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #0f0;
            font-size: 14px;
        }
        
        .stat-value {
            color: #0f0;
            font-weight: bold;
        }
        
        select {
            padding: 8px;
            font-size: 14px;
            background: #222;
            color: #fff;
            border: 1px solid #0f0;
            border-radius: 5px;
        }
        
        .preset-buttons {
            display: flex;
            gap: 10px;
        }
        
        .preset-btn {
            padding: 8px 15px;
            font-size: 14px;
            background: #444;
            color: #fff;
            border: 1px solid #666;
            cursor: pointer;
        }
        
        .preset-btn.active {
            background: #0f0;
            color: #000;
            border-color: #0f0;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    
    <div class="stats">
        <div>FPS: <span id="fps" class="stat-value">0</span></div>
        <div>Processing: <span id="latency" class="stat-value">0</span>ms</div>
        <div>Frames: <span id="totalFrames" class="stat-value">0</span></div>
        <div>Skip Rate: <span id="skipRate" class="stat-value">0</span>%</div>
    </div>
    
    <div class="controls">
        <button id="startBtn" onclick="start()">🚀 Start</button>
        <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
        
        <div class="preset-buttons">
            <button class="preset-btn active" onclick="setPreset('fast')" id="fast">Fast (0.4)</button>
            <button class="preset-btn" onclick="setPreset('balanced')" id="balanced">Balanced (0.6)</button>
            <button class="preset-btn" onclick="setPreset('quality')" id="quality">Quality (0.8)</button>
        </div>
        
        <select id="styleSelect">
            <option value="">None</option>
            <option value="cyberpunk">Cyberpunk</option>
            <option value="anime">Anime</option>
            <option value="oil painting">Oil Painting</option>
            <option value="watercolor">Watercolor</option>
            <option value="sketch">Sketch</option>
        </select>
        
        <label>Strength: <span id="strengthValue">0.4</span></label>
        <input type="range" id="strengthSlider" min="0.3" max="0.9" step="0.05" value="0.4" oninput="updateStrength()">
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        let sendInterval = null;
        
        function updateStrength() {
            document.getElementById('strengthValue').textContent = 
                document.getElementById('strengthSlider').value;
        }
        
        function setPreset(preset) {
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(preset).classList.add('active');
            
            const slider = document.getElementById('strengthSlider');
            if (preset === 'fast') slider.value = 0.4;
            else if (preset === 'balanced') slider.value = 0.6;
            else if (preset === 'quality') slider.value = 0.8;
            
            updateStrength();
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
                
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Connected');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    
                    // Enviar frames a 15 FPS
                    sendInterval = setInterval(sendFrame, 66);
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
                
                ws.onerror = () => stop();
                ws.onclose = () => stop();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const video = document.getElementById('video');
            if (!video.videoWidth || !video.videoHeight) return;
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Center crop
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            const style = document.getElementById('styleSelect').value;
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.8),
                prompt: style || "photorealistic",
                strength: strength,
                timestamp: Date.now()
            }));
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
            
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skipRate').textContent = 
                    data.stats.skip_rate.toFixed(1);
            }
        }
        
        function stop() {
            streaming = false;
            if (sendInterval) clearInterval(sendInterval);
            if (ws) ws.close();
            
            const video = document.getElementById('video');
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
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
    print("Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"Error decodificando imagen: {e}")
                continue
            
            processor.process_frame(
                input_image, 
                data.get('prompt', 'photorealistic'),
                data.get('strength', 0.5),
                data.get('timestamp', time.time() * 1000)
            )
            
            result = processor.get_latest_result()
            
            if result:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp'],
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🚀 FAST LCM STREAMDIFFUSION")
    print("="*60)
    print("⚡ LCM Dreamshaper v7 (1-step inference)")
    print("🎯 Optimized for 15-25 FPS")
    print("⏩ Asynchronous processing")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
