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

class OptimizedStreamDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Frame skipping
        self.skip_frames = 2  # Process every 3rd frame
        self.frame_counter = 0
        
        # Performance metrics
        self.total_frames = 0
        self.processed_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando Optimized StreamDiffusion...")
        
        # Usar SD-Turbo para máxima velocidad
        model_id = "stabilityai/sd-turbo"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        )
        
        self.pipe = self.pipe.to(self.device)
        
        # Configurar scheduler LCM para velocidad
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
                        guidance_scale=1.0  # SD-Turbo necesita al menos 1.0
                    ).images[0]
        except Exception as e:
            print(f"⚠ Pre-calentamiento falló, continuando: {e}")
        
        print("✓ Optimized StreamDiffusion listo!")
        self.start_processing_thread()
    
    def start_processing_thread(self):
        self.processing = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
    
    def _processing_loop(self):
        """Loop optimizado para máxima velocidad"""
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Generación ultra-rápida
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        result = self.pipe(
                            prompt=frame_data['prompt'],
                            image=frame_data['image'],
                            num_inference_steps=1,  # Solo 1 paso!
                            strength=frame_data.get('strength', 0.5),
                            guidance_scale=1.0,  # SD-Turbo mínimo 1.0
                            generator=torch.Generator(device=self.device).manual_seed(42)
                        ).images[0]
                
                processing_time = (time.time() - start_time) * 1000
                self.processed_frames += 1
                
                result_data = {
                    'image': result,
                    'processing_time': processing_time,
                    'timestamp': frame_data['timestamp'],
                    'stats': {
                        'total_frames': self.total_frames,
                        'processed_frames': self.processed_frames,
                        'skip_rate': (1 - self.processed_frames / max(1, self.total_frames)) * 100
                    }
                }
                
                try:
                    self.result_queue.put_nowait(result_data)
                except queue.Full:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result_data)
                
                print(f"Frame procesado: {processing_time:.1f}ms")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error: {e}")
                continue
    
    def add_frame(self, image, prompt, strength, timestamp):
        self.total_frames += 1
        self.frame_counter += 1
        
        # Skip frames para mayor velocidad
        if self.frame_counter % (self.skip_frames + 1) != 0:
            return
        
        frame_data = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'timestamp': timestamp
        }
        
        try:
            self.frame_queue.put_nowait(frame_data)
        except queue.Full:
            self.frame_queue.get_nowait()
            self.frame_queue.put_nowait(frame_data)
    
    def get_latest_result(self):
        result = None
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
        return result

# Instancia global
processor = OptimizedStreamDiffusion()

# HTML simplificado para mayor rendimiento
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Optimized StreamDiffusion - Ultra Fast</title>
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
        
        <select id="styleSelect">
            <option value="">Photorealistic</option>
            <option value="cyberpunk">Cyberpunk</option>
            <option value="anime">Anime</option>
            <option value="oil painting">Oil Painting</option>
            <option value="watercolor">Watercolor</option>
        </select>
        
        <label>Strength: <span id="strengthValue">0.5</span></label>
        <input type="range" id="strengthSlider" min="0.3" max="0.7" step="0.05" value="0.5" oninput="updateStrength()">
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        function updateStrength() {
            document.getElementById('strengthValue').textContent = 
                document.getElementById('strengthSlider').value;
        }
        
        async function start() {
            try {
                const video = document.getElementById('video');
                const canvas = document.getElementById('output');
                const ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 512, height: 512 }
                });
                video.srcObject = stream;
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Connected');
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
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 512, 512);
            
            const style = document.getElementById('styleSelect').value;
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.9),
                prompt: style || "photorealistic",
                strength: strength,
                timestamp: Date.now()
            }));
            
            setTimeout(sendFrame, 33); // ~30 FPS input
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
            
            processor.add_frame(
                input_image, 
                data.get('prompt', 'photorealistic'),
                data.get('strength', 0.5),
                data.get('timestamp', time.time() * 1000)
            )
            
            result = processor.get_latest_result()
            
            if result:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=90)
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
    print("🚀 OPTIMIZED STREAMDIFFUSION - ULTRA FAST MODE")
    print("="*60)
    print("⚡ SD-Turbo model for maximum speed")
    print("🎯 1-step inference")
    print("⏩ Frame skipping enabled")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
