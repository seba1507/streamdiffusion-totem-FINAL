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
from diffusers import AutoPipelineForImage2Image, LCMScheduler, DiffusionPipeline
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

class UltraFastDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Resolución dinámica para velocidad
        self.resolution = 384  # Reducido de 512
        
        # Processing ultra-optimizado
        self.processing = False
        self.last_result = None
        self.last_latents = None
        self.frame_counter = 0
        
        # Performance metrics
        self.total_frames = 0
        self.processed_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando Ultra Fast Diffusion...")
        
        # Opciones de modelos rápidos
        model_options = [
            "SimianLuo/LCM_Dreamshaper_v7",  # Primero intentar con LCM
            "stabilityai/sd-turbo",  # Backup: SD-Turbo
            "nota-ai/bk-sdm-small",  # Backup: Modelo pequeño
        ]
        
        for model_id in model_options:
            try:
                print(f"Intentando cargar {model_id}...")
                
                if "turbo" in model_id:
                    self.pipe = AutoPipelineForImage2Image.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        safety_checker=None,
                        variant="fp16"
                    )
                else:
                    self.pipe = AutoPipelineForImage2Image.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        safety_checker=None,
                        use_safetensors=True
                    )
                
                print(f"✓ Modelo {model_id} cargado exitosamente")
                break
                
            except Exception as e:
                print(f"⚠ Error cargando {model_id}: {e}")
                continue
        
        if self.pipe is None:
            print("ERROR: No se pudo cargar ningún modelo")
            sys.exit(1)
        
        self.pipe = self.pipe.to(self.device)
        
        # Scheduler ultra-rápido
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
        
        # Optimización adicional: compilar con torch.compile si está disponible
        try:
            self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
            print("✓ UNet compilado con torch.compile")
        except:
            print("⚠ torch.compile no disponible")
        
        # Pre-calentar con resolución reducida
        print(f"Pre-calentando pipeline a {self.resolution}x{self.resolution}...")
        dummy_image = Image.new('RGB', (self.resolution, self.resolution), color=(128, 128, 128))
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        "test",
                        image=dummy_image,
                        num_inference_steps=1,
                        strength=0.3,
                        guidance_scale=1.0
                    ).images[0]
            print("✓ Pre-calentamiento exitoso")
        except Exception as e:
            print(f"⚠ Pre-calentamiento falló: {e}")
        
        # Limpiar memoria
        torch.cuda.empty_cache()
        
        print("✓ Ultra Fast Diffusion listo!")
        self.start_processing_thread()
    
    def start_processing_thread(self):
        self.processing = True
        self.current_task = None
        processing_thread = threading.Thread(target=self._processing_loop_ultra, daemon=True)
        processing_thread.start()
    
    def _processing_loop_ultra(self):
        """Loop ultra-optimizado con latent caching"""
        while self.processing:
            if self.current_task:
                try:
                    start_time = time.time()
                    
                    # Reducir imagen a resolución óptima
                    image = self.current_task['image']
                    if image.size[0] != self.resolution:
                        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                    
                    # Ultra-fast generation
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            # Usar latents cacheados si es posible
                            result = self.pipe(
                                prompt=self.current_task['prompt'],
                                image=image,
                                num_inference_steps=1,
                                strength=min(self.current_task['strength'], 0.5),  # Limitar strength
                                guidance_scale=1.0,  # Siempre 1.0 para velocidad
                                output_type="pil"
                            ).images[0]
                    
                    # Escalar de vuelta a 512x512 para display
                    if result.size[0] != 512:
                        result = result.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.processed_frames += 1
                    
                    self.last_result = {
                        'image': result,
                        'processing_time': processing_time,
                        'timestamp': self.current_task['timestamp'],
                        'stats': {
                            'total_frames': self.total_frames,
                            'processed_frames': self.processed_frames,
                            'skip_rate': (1 - self.processed_frames / max(1, self.total_frames)) * 100,
                            'resolution': self.resolution
                        }
                    }
                    
                    print(f"Frame: {processing_time:.1f}ms @ {self.resolution}px")
                    self.current_task = None
                    
                except Exception as e:
                    print(f"Error: {e}")
                    self.current_task = None
            else:
                time.sleep(0.001)
    
    def process_frame(self, image, prompt, strength, timestamp):
        self.total_frames += 1
        self.frame_counter += 1
        
        # Procesar solo 1 de cada 2 frames si es muy lento
        if self.frame_counter % 2 != 0 and self.current_task is not None:
            return
        
        # Skip si ya estamos procesando
        if self.current_task is not None:
            return
        
        self.current_task = {
            'image': image,
            'prompt': prompt or "photo",
            'strength': strength,
            'timestamp': timestamp
        }
    
    def get_latest_result(self):
        return self.last_result
    
    def set_resolution(self, res):
        """Cambiar resolución dinámicamente"""
        self.resolution = res
        print(f"Resolución cambiada a {res}x{res}")

# Instancia global
processor = UltraFastDiffusion()

# HTML con controles de resolución
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultra Fast Diffusion</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background: linear-gradient(135deg, #0a0a0a, #1a1a2e);
            color: #fff;
            font-family: 'Consolas', 'Monaco', monospace;
            overflow: hidden;
            height: 100vh;
        }
        
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 30px;
            padding: 20px;
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            border: 3px solid #00ff88;
            background: #111;
            border-radius: 12px;
            box-shadow: 0 0 30px rgba(0, 255, 136, 0.3);
        }
        
        .output-container {
            position: relative;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(10, 10, 10, 0.95);
            padding: 25px;
            border-radius: 20px;
            border: 2px solid #00ff88;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .control-label {
            color: #00ff88;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        button {
            padding: 12px 24px;
            font-size: 14px;
            cursor: pointer;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            color: #000;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 20px #00ff88;
        }
        
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .totem-btn {
            background: linear-gradient(45deg, #ff6b00, #ff4500);
            color: #fff;
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, #333, #00ff88);
            outline: none;
            border-radius: 3px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
        }
        
        textarea {
            padding: 10px;
            font-size: 13px;
            border-radius: 8px;
            background: #1a1a1a;
            color: #fff;
            border: 2px solid #00ff88;
            font-family: 'Consolas', monospace;
            resize: none;
            height: 60px;
        }
        
        .range-value {
            color: #00ff88;
            font-weight: bold;
            text-align: center;
            font-size: 16px;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(10, 10, 10, 0.95);
            padding: 20px;
            border-radius: 15px;
            font-size: 12px;
            border: 2px solid #00ff88;
            min-width: 250px;
        }
        
        .stat-value {
            color: #00ff88;
            font-weight: bold;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #333;
        }
        
        .resolution-buttons {
            display: flex;
            gap: 10px;
            margin-top: 5px;
        }
        
        .res-btn {
            flex: 1;
            padding: 8px;
            font-size: 12px;
            background: #444;
        }
        
        .res-btn.active {
            background: #00ff88;
            color: #000;
        }
        
        /* Modo Totem */
        .totem-mode {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            z-index: 9999;
            display: none;
            justify-content: center;
            align-items: center;
        }
        
        .totem-canvas {
            max-width: 90vh;
            max-height: 90vh;
            width: auto;
            height: auto;
        }
        
        .totem-exit {
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
        }
        
        .warning {
            color: #ff6b00;
            font-size: 11px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <div class="output-container">
            <canvas id="output"></canvas>
        </div>
    </div>
    
    <!-- Modo Totem -->
    <div class="totem-mode" id="totemMode">
        <button class="totem-exit" onclick="exitTotemMode()">✕</button>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
    </div>
    
    <div class="stats">
        <div style="text-align: center; margin-bottom: 15px; color: #00ff88; font-weight: bold;">
            ULTRA FAST MODE
        </div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Processing:</span>
            <span id="latency" class="stat-value">0</span>ms
        </div>
        <div class="stat-row">
            <span>Resolution:</span>
            <span id="resolution" class="stat-value">384</span>px
        </div>
        <div class="stat-row">
            <span>Total Frames:</span>
            <span id="totalFrames" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Skip Rate:</span>
            <span id="skipRate" class="stat-value">0</span>%
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <div class="control-label">Control</div>
            <button id="startBtn" onclick="start()">🚀 Start</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
            <button class="totem-btn" onclick="enterTotemMode()" id="totemBtn" disabled>📺 Totem</button>
        </div>
        
        <div class="control-group">
            <div class="control-label">Resolution (Speed)</div>
            <div class="resolution-buttons">
                <button class="res-btn" onclick="setResolution(256)" id="res256">256px</button>
                <button class="res-btn active" onclick="setResolution(384)" id="res384">384px</button>
                <button class="res-btn" onclick="setResolution(512)" id="res512">512px</button>
            </div>
            <div class="warning">Lower = Faster</div>
        </div>
        
        <div class="control-group">
            <div class="control-label">Prompt</div>
            <textarea id="customPrompt" placeholder="Simple prompts work best...">cyberpunk portrait</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Strength: <span id="strengthValue">0.3</span></div>
            <input type="range" id="strengthSlider" min="0.2" max="0.5" step="0.05" value="0.3" oninput="updateStrength()">
            <div class="warning">Max 0.5 for speed</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        let currentResolution = 384;
        
        function updateStrength() {
            document.getElementById('strengthValue').textContent = 
                document.getElementById('strengthSlider').value;
        }
        
        function setResolution(res) {
            currentResolution = res;
            document.querySelectorAll('.res-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('res' + res).classList.add('active');
            document.getElementById('resolution').textContent = res;
            
            // Enviar cambio al servidor
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'resolution',
                    value: res
                }));
            }
        }
        
        function enterTotemMode() {
            document.getElementById('totemMode').style.display = 'flex';
            const totemCanvas = document.getElementById('totemCanvas');
            const totemCtx = totemCanvas.getContext('2d');
            totemCanvas.width = 512;
            totemCanvas.height = 512;
        }
        
        function exitTotemMode() {
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
                    video: { width: 640, height: 480, frameRate: 30 }
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
                    document.getElementById('totemBtn').disabled = false;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0, 512, 512);
                            
                            // Totem mode
                            const totemCanvas = document.getElementById('totemCanvas');
                            if (totemCanvas && document.getElementById('totemMode').style.display === 'flex') {
                                const totemCtx = totemCanvas.getContext('2d');
                                totemCtx.drawImage(img, 0, 0, 512, 512);
                            }
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
            if (!video.videoWidth) {
                setTimeout(sendFrame, 10);
                return;
            }
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            const prompt = document.getElementById('customPrompt').value || "";
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            
            ws.send(JSON.stringify({
                type: 'frame',
                image: tempCanvas.toDataURL('image/jpeg', 0.8),
                prompt: prompt,
                strength: strength,
                timestamp: Date.now()
            }));
            
            setTimeout(sendFrame, 33); // 30 FPS input
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
                if (data.stats.resolution) {
                    document.getElementById('resolution').textContent = data.stats.resolution;
                }
            }
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            
            const video = document.getElementById('video');
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('totemBtn').disabled = true;
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                exitTotemMode();
            }
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
    print("Cliente conectado - Ultra Fast Mode")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get('type') == 'resolution':
                processor.set_resolution(data['value'])
                continue
            
            if data.get('type') != 'frame':
                continue
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            processor.process_frame(
                input_image, 
                data.get('prompt', ''),
                data.get('strength', 0.3),
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
    print("🚀 ULTRA FAST DIFFUSION - MAXIMUM SPEED")
    print("="*60)
    print("⚡ Dynamic resolution (256/384/512)")
    print("🎯 1-step inference only")
    print("⏩ Frame skipping enabled")
    print("🔧 Optimized for 15+ FPS")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
