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

# Optimizaciones para A6000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.cuda.set_per_process_memory_fraction(0.9)

# Desactivar torch.compile que está causando problemas
torch._dynamo.config.suppress_errors = True

# Verificar CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

class A6000OptimizedDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Configuración optimizada para A6000
        self.resolution = 320  # Resolución intermedia optimizada
        self.batch_size = 1
        
        # Queues optimizadas
        self.current_task = None
        self.last_result = None
        self.processing = False
        
        # Metrics
        self.total_frames = 0
        self.processed_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando A6000 Optimized Diffusion...")
        
        # Probar diferentes modelos en orden de velocidad
        model_options = [
            ("nota-ai/bk-sdm-small", "small"),  # Modelo más pequeño
            ("SimianLuo/LCM_Dreamshaper_v7", "lcm"),  # LCM
            ("stabilityai/sd-turbo", "turbo"),  # SD-Turbo
        ]
        
        for model_id, model_type in model_options:
            try:
                print(f"Cargando {model_id}...")
                
                # Configuración específica por modelo
                if model_type == "small":
                    self.pipe = DiffusionPipeline.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        safety_checker=None,
                        custom_pipeline="latent_consistency_img2img",
                    )
                else:
                    self.pipe = AutoPipelineForImage2Image.from_pretrained(
                        model_id,
                        torch_dtype=self.dtype,
                        safety_checker=None,
                        use_safetensors=True,
                        variant="fp16" if "turbo" in model_type else None
                    )
                
                print(f"✓ Modelo {model_id} cargado")
                break
                
            except Exception as e:
                print(f"⚠ Error con {model_id}: {e}")
                continue
        
        self.pipe = self.pipe.to(self.device)
        
        # LCM Scheduler para todos los modelos
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        
        # Optimizaciones críticas
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers es crítico para A6000
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ XFormers habilitado - CRÍTICO para rendimiento")
        except Exception as e:
            print(f"❌ ERROR: XFormers falló: {e}")
            print("⚠️  El rendimiento será MUCHO menor sin XFormers")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado con slicing y tiling")
        except:
            pass
        
        # Optimización adicional: channels_last memory format
        try:
            self.pipe.unet = self.pipe.unet.to(memory_format=torch.channels_last)
            self.pipe.vae = self.pipe.vae.to(memory_format=torch.channels_last)
            print("✓ Memory format optimizado (channels_last)")
        except:
            pass
        
        # Pre-calentar
        print(f"Pre-calentando a {self.resolution}x{self.resolution}...")
        dummy_image = Image.new('RGB', (self.resolution, self.resolution), color=(128, 128, 128))
        
        for i in range(3):  # 3 warmup runs
            try:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(dtype=self.dtype):
                        _ = self.pipe(
                            prompt="test",
                            image=dummy_image,
                            num_inference_steps=1,
                            strength=0.3,
                            guidance_scale=1.0,
                            output_type="pil"
                        ).images[0]
                print(f"✓ Warmup {i+1}/3")
            except Exception as e:
                print(f"⚠ Warmup {i+1} falló: {e}")
        
        # Limpiar cache después del warmup
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        print("✓ A6000 Optimized Diffusion listo!")
        self.start_processing_thread()
    
    def start_processing_thread(self):
        self.processing = True
        thread = threading.Thread(target=self._processing_loop, daemon=True)
        thread.start()
    
    def _processing_loop(self):
        """Loop optimizado para A6000"""
        while self.processing:
            if self.current_task is not None:
                try:
                    start_time = time.time()
                    
                    # Preparar imagen
                    image = self.current_task['image']
                    if image.size[0] != self.resolution:
                        image = image.resize((self.resolution, self.resolution), Image.Resampling.LANCZOS)
                    
                    # Generación optimizada
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=self.dtype):
                            result = self.pipe(
                                prompt=self.current_task['prompt'] or "",
                                image=image,
                                num_inference_steps=1,
                                strength=min(self.current_task['strength'], 0.4),
                                guidance_scale=1.0,
                                output_type="pil",
                                return_dict=True
                            ).images[0]
                    
                    # Sincronizar GPU
                    torch.cuda.synchronize()
                    
                    # Escalar resultado
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
                    
                    # Limpiar cache periódicamente
                    if self.processed_frames % 50 == 0:
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"Error procesando: {e}")
                    self.current_task = None
                    torch.cuda.empty_cache()
            else:
                time.sleep(0.001)
    
    def process_frame(self, image, prompt, strength, timestamp):
        self.total_frames += 1
        
        # Skip si estamos procesando
        if self.current_task is not None:
            return
        
        self.current_task = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'timestamp': timestamp
        }
    
    def get_latest_result(self):
        return self.last_result
    
    def set_resolution(self, res):
        self.resolution = res
        torch.cuda.empty_cache()
        print(f"Resolución: {res}x{res}")

# Instancia global
processor = A6000OptimizedDiffusion()

# HTML optimizado para A6000
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>A6000 Optimized Diffusion</title>
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
        
        .status-indicator {
            position: absolute;
            top: 15px;
            left: 15px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4444;
            transition: all 0.3s;
        }
        
        .status-indicator.active {
            background: #00ff88;
            box-shadow: 0 0 15px #00ff88;
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
            gap: 25px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .control-label {
            color: #00ff88;
            font-weight: bold;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 5px;
        }
        
        button {
            padding: 15px 30px;
            font-size: 16px;
            cursor: pointer;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            color: #000;
            border: none;
            border-radius: 10px;
            font-weight: bold;
            transition: all 0.3s;
            text-transform: uppercase;
        }
        
        button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px #00ff88;
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
            height: 8px;
            -webkit-appearance: none;
            background: linear-gradient(90deg, #333, #00ff88);
            outline: none;
            border-radius: 4px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 24px;
            height: 24px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        }
        
        textarea {
            padding: 15px;
            font-size: 14px;
            border-radius: 10px;
            background: #1a1a1a;
            color: #fff;
            border: 2px solid #00ff88;
            font-family: 'Consolas', monospace;
            resize: vertical;
            min-height: 80px;
            max-height: 120px;
        }
        
        .range-value {
            color: #00ff88;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            margin-top: 5px;
            text-shadow: 0 0 10px #00ff88;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(10, 10, 10, 0.95);
            padding: 25px;
            border-radius: 20px;
            font-family: 'Consolas', monospace;
            font-size: 12px;
            border: 2px solid #00ff88;
            min-width: 280px;
            backdrop-filter: blur(10px);
        }
        
        .stat-value {
            color: #00ff88;
            font-weight: bold;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            border-bottom: 1px solid #333;
            padding-bottom: 8px;
        }
        
        .title {
            text-align: center;
            margin-bottom: 20px;
            color: #00ff88;
            font-size: 16px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
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
            border: none;
            box-shadow: 0 0 50px rgba(0, 255, 136, 0.5);
        }
        
        .totem-exit {
            position: absolute;
            top: 30px;
            right: 30px;
            background: rgba(255, 0, 0, 0.8);
            color: white;
            border: none;
            border-radius: 50%;
            width: 70px;
            height: 70px;
            font-size: 28px;
            cursor: pointer;
            z-index: 10000;
        }
        
        .totem-info {
            position: absolute;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.9);
            padding: 20px 40px;
            border-radius: 15px;
            border: 2px solid #00ff88;
            text-align: center;
        }
        
        .resolution-buttons {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .res-btn {
            flex: 1;
            padding: 10px;
            font-size: 14px;
            background: #444;
            border: 1px solid #666;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
        }
        
        .res-btn.active {
            background: #00ff88;
            color: #000;
            border-color: #00ff88;
        }
        
        .performance-info {
            color: #999;
            font-size: 11px;
            margin-top: 5px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <div class="output-container">
            <canvas id="output"></canvas>
            <div class="status-indicator" id="statusIndicator"></div>
        </div>
    </div>
    
    <!-- Modo Totem -->
    <div class="totem-mode" id="totemMode">
        <button class="totem-exit" onclick="exitTotemMode()">✕</button>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
        <div class="totem-info">
            <div style="color: #00ff88; font-size: 20px; font-weight: bold;">A6000 OPTIMIZED MODE</div>
            <div style="margin-top: 10px;">RTX A6000 - 48GB VRAM</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">A6000 Performance</div>
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
            <span id="resolution" class="stat-value">320</span>px
        </div>
        <div class="stat-row">
            <span>Total Frames:</span>
            <span id="totalFrames" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Skip Rate:</span>
            <span id="skipRate" class="stat-value">0</span>%
        </div>
        <div class="stat-row">
            <span>Strength:</span>
            <span id="currentStrength" class="stat-value">0.3</span>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <div class="control-label">System Control</div>
            <button id="startBtn" onclick="start()">🚀 Start Stream</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
            <button class="totem-btn" onclick="enterTotemMode()" id="totemBtn" disabled>📺 Totem Mode</button>
        </div>
        
        <div class="control-group">
            <div class="control-label">Resolution</div>
            <div class="resolution-buttons">
                <button class="res-btn" onclick="setResolution(256)" id="res256">256px</button>
                <button class="res-btn active" onclick="setResolution(320)" id="res320">320px</button>
                <button class="res-btn" onclick="setResolution(384)" id="res384">384px</button>
            </div>
            <div class="performance-info">Lower = Faster | 320px recommended</div>
        </div>
        
        <div class="control-group">
            <div class="control-label">Custom Prompt</div>
            <textarea id="customPrompt" placeholder="Enter your prompt (simple = faster)">cyberpunk portrait</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Strength: <span id="strengthValue">0.3</span></div>
            <input type="range" id="strengthSlider" min="0.2" max="0.4" step="0.05" value="0.3" oninput="updateStrengthValue()">
            <div class="performance-info">Limited to 0.4 for optimal speed</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        let canvas = null;
        let ctx = null;
        let totemCanvas = null;
        let totemCtx = null;
        let totemMode = false;
        let currentResolution = 320;
        
        let frameCount = 0;
        let totalFrames = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
            document.getElementById('currentStrength').textContent = value.toFixed(2);
        }
        
        function setResolution(res) {
            currentResolution = res;
            document.querySelectorAll('.res-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('res' + res).classList.add('active');
            document.getElementById('resolution').textContent = res;
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'resolution',
                    value: res
                }));
            }
        }
        
        function enterTotemMode() {
            if (!streaming) {
                alert('Start the stream before activating totem mode');
                return;
            }
            
            totemMode = true;
            document.getElementById('totemMode').style.display = 'flex';
            
            totemCanvas = document.getElementById('totemCanvas');
            totemCtx = totemCanvas.getContext('2d');
            
            const screenHeight = window.innerHeight;
            const screenWidth = window.innerWidth;
            
            if (screenHeight > screenWidth) {
                totemCanvas.width = screenWidth * 0.8;
                totemCanvas.height = screenWidth * 0.8;
            } else {
                totemCanvas.width = screenHeight * 0.7;
                totemCanvas.height = screenHeight * 0.7;
            }
            
            document.querySelector('.controls').style.display = 'none';
            document.querySelector('.stats').style.display = 'none';
            
            console.log('Totem Mode activated');
        }
        
        function exitTotemMode() {
            totemMode = false;
            document.getElementById('totemMode').style.display = 'none';
            document.querySelector('.controls').style.display = 'grid';
            document.querySelector('.stats').style.display = 'block';
            console.log('Totem Mode deactivated');
        }
        
        async function start() {
            try {
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d');
                canvas.width = 512;
                canvas.height = 512;
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    }
                });
                video.srcObject = stream;
                
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Connected to A6000 Optimized server');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    document.getElementById('totemBtn').disabled = false;
                    document.getElementById('statusIndicator').classList.add('active');
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, 512, 512);
                            ctx.drawImage(img, 0, 0, 512, 512);
                            
                            if (totemMode && totemCtx) {
                                totemCtx.clearRect(0, 0, totemCanvas.width, totemCanvas.height);
                                totemCtx.drawImage(img, 0, 0, totemCanvas.width, totemCanvas.height);
                            }
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
                    console.log('Disconnected from server');
                    stop();
                };
                
            } catch (error) {
                console.error('Error:', error);
                alert('Error: ' + error.message);
            }
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                setTimeout(sendFrame, 10);
                return;
            }
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Center crop
            const videoAspect = video.videoWidth / video.videoHeight;
            const targetAspect = 1;
            
            let sx, sy, sw, sh;
            
            if (videoAspect > targetAspect) {
                sh = video.videoHeight;
                sw = video.videoHeight;
                sx = (video.videoWidth - sw) / 2;
                sy = 0;
            } else {
                sw = video.videoWidth;
                sh = video.videoWidth;
                sx = 0;
                sy = (video.videoHeight - sh) / 2;
            }
            
            tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
            
            const prompt = document.getElementById('customPrompt').value || "portrait";
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.85);
            ws.send(JSON.stringify({
                type: 'frame',
                image: imageData,
                prompt: prompt,
                strength: strength,
                timestamp: Date.now()
            }));
            
            // Send frames at ~20 FPS
            setTimeout(sendFrame, 50);
        }
        
        function updateStats(data) {
            frameCount++;
            totalFrames++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 50) latencies.shift();
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
            
            if (totemMode) {
                exitTotemMode();
            }
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            if (ctx) {
                ctx.clearRect(0, 0, 512, 512);
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('totemBtn').disabled = true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape' && totemMode) {
                exitTotemMode();
            }
            if (event.key === 'F11') {
                event.preventDefault();
                if (!totemMode && streaming) {
                    enterTotemMode();
                }
            }
        });
        
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
    print("🚀 Cliente conectado - A6000 Optimized Mode")
    
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
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            prompt = data.get('prompt', 'portrait')
            strength = data.get('strength', 0.3)
            
            processor.process_frame(
                input_image, 
                prompt,
                strength,
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
        print(f"❌ Error en WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎨 A6000 OPTIMIZED DIFFUSION")
    print("="*80)
    print("⚡ RTX A6000 - 48GB VRAM")
    print("🎯 Optimizaciones específicas para Ampere")
    print("📐 Resolución dinámica (256/320/384)")
    print("🔧 XFormers + TF32 + Channels Last")
    print("🎮 Custom Prompts + Totem Mode")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
