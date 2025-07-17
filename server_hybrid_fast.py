#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import hashlib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
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

class HybridDotSimulateDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Processing queues optimizadas
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Frame skipping inteligente
        self.last_image_hash = None
        self.similarity_threshold = 0.85  # Más agresivo para velocidad
        self.frame_skip_counter = 0
        
        # Temporal feedback system (reduce saltiness)
        self.frame_buffer = deque(maxlen=2)  # Menos frames para más velocidad
        self.temporal_smoothing = 0.1  # Menos smoothing para menor latencia
        
        # Performance metrics
        self.total_frames = 0
        self.skipped_frames = 0
        self.preprocessing_time = 0
        
        # DotSimulate prompt enhancement
        self.quality_boosters = [
            "highly detailed", "professional", "sharp focus",
            "high quality", "detailed textures"
        ]
        
        self.init_model()
        
    def init_model(self):
        print("Inicializando Hybrid DotSimulate Diffusion...")
        
        # LCM Dreamshaper para balance velocidad/calidad
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
        except Exception as e:
            print(f"⚠ XFormers no disponible: {e}")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE optimizado")
        except Exception as e:
            print(f"⚠ Optimizaciones VAE fallaron: {e}")
        
        # Pre-calentamiento
        print("Pre-calentando pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        try:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _ = self.pipe(
                        "test",
                        image=dummy_image,
                        num_inference_steps=2,
                        strength=0.5,
                        guidance_scale=1.0
                    ).images[0]
            print("✓ Pre-calentamiento exitoso")
        except Exception as e:
            print(f"⚠ Error en pre-calentamiento: {e}")
        
        print("✓ Hybrid DotSimulate Diffusion listo!")
        self.start_processing_thread()
    
    def preprocess_image_fast(self, image):
        """
        Preprocessing optimizado para velocidad manteniendo calidad
        """
        start_time = time.time()
        
        # 1. Ajuste rápido de contraste y color
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.15)
        
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        # 2. Edge enhancement simplificado
        img_array = np.array(image)
        
        # 3. Noise ligero para textura (más rápido)
        noise = np.random.normal(0, 1.5, img_array.shape)
        img_array = img_array.astype(np.float32) + noise
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        image = Image.fromarray(img_array)
        
        self.preprocessing_time = (time.time() - start_time) * 1000
        return image
    
    def enhance_prompt_fast(self, base_prompt):
        """
        Prompt enhancement simplificado para velocidad
        """
        if not base_prompt.strip():
            return "professional photo"
        
        # Solo agregar 2-3 quality boosters para no ralentizar
        quality_terms = self.quality_boosters[:3]
        return f"{base_prompt}, {', '.join(quality_terms)}"
    
    def calculate_image_similarity(self, image):
        """Hash rápido para similarity"""
        try:
            small_img = image.resize((16, 16))  # Más pequeño = más rápido
            img_array = np.array(small_img)
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except:
            return str(time.time())
    
    def should_skip_frame(self, image):
        """Frame skipping agresivo para velocidad"""
        try:
            current_hash = self.calculate_image_similarity(image)
            
            if self.last_image_hash is None:
                self.last_image_hash = current_hash
                return False
            
            if current_hash == self.last_image_hash:
                self.frame_skip_counter += 1
                return True
            
            self.last_image_hash = current_hash
            self.frame_skip_counter = 0
            return False
        except:
            return False
    
    def apply_temporal_smoothing(self, generated_image):
        """Smoothing mínimo para reducir latencia"""
        if len(self.frame_buffer) == 0:
            self.frame_buffer.append(generated_image)
            return generated_image
        
        try:
            current_array = np.array(generated_image).astype(np.float32)
            previous_array = np.array(self.frame_buffer[-1]).astype(np.float32)
            
            # Smoothing muy ligero
            smoothed = (current_array * 0.9 + previous_array * 0.1)
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
            smoothed_image = Image.fromarray(smoothed)
            
            self.frame_buffer.append(smoothed_image)
            return smoothed_image
        except:
            self.frame_buffer.append(generated_image)
            return generated_image
    
    def calculate_optimal_steps(self, strength, guidance_scale):
        """Pasos optimizados para velocidad"""
        if strength >= 0.7:
            return 2  # Máximo 2 pasos
        else:
            return 1  # 1 paso para máxima velocidad
    
    def start_processing_thread(self):
        self.processing = True
        processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        processing_thread.start()
        print("✓ Processing thread iniciado")
    
    def _processing_loop(self):
        """Loop optimizado para máxima velocidad con calidad"""
        while self.processing:
            try:
                frame_data = self.frame_queue.get(timeout=0.05)  # Timeout más corto
                
                start_time = time.time()
                
                # Skip frames similares
                if self.should_skip_frame(frame_data['image']):
                    self.skipped_frames += 1
                    # Reusar último resultado
                    try:
                        if self.result_queue.qsize() > 0:
                            last_result = list(self.result_queue.queue)[-1]
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                # Preprocessing rápido
                processed_image = self.preprocess_image_fast(frame_data['image'])
                
                # Prompt enhancement
                enhanced_prompt = self.enhance_prompt_fast(frame_data['prompt'])
                
                # Parámetros optimizados
                strength = frame_data.get('strength', 0.5)
                guidance_scale = frame_data.get('guidance_scale', 1.0)  # Bajo para velocidad
                steps = self.calculate_optimal_steps(strength, guidance_scale)
                
                # Generación rápida
                try:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            result = self.pipe(
                                prompt=enhanced_prompt,
                                image=processed_image,
                                num_inference_steps=steps,
                                strength=strength,
                                guidance_scale=guidance_scale,
                                generator=torch.Generator(device=self.device).manual_seed(42)
                            ).images[0]
                    
                    # Temporal smoothing ligero
                    result = self.apply_temporal_smoothing(result)
                    
                    processing_time = (time.time() - start_time) * 1000
                    self.total_frames += 1
                    
                    result_data = {
                        'image': result,
                        'processing_time': processing_time,
                        'preprocessing_time': self.preprocessing_time,
                        'timestamp': frame_data['timestamp'],
                        'enhanced_prompt': enhanced_prompt,
                        'stats': {
                            'total_frames': self.total_frames,
                            'skipped_frames': self.skipped_frames,
                            'skip_rate': self.skipped_frames / max(1, self.total_frames) * 100,
                            'strength': strength,
                            'steps': steps,
                            'guidance_scale': guidance_scale,
                            'temporal_smoothing': self.temporal_smoothing
                        }
                    }
                    
                    try:
                        self.result_queue.put_nowait(result_data)
                    except queue.Full:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_data)
                    
                    print(f"Frame: {processing_time:.1f}ms (steps: {steps})")
                    
                except Exception as e:
                    print(f"Error procesando: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en loop: {e}")
    
    def add_frame(self, image, prompt, strength, guidance_scale, timestamp):
        frame_data = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
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
processor = HybridDotSimulateDiffusion()

# HTML con todas las características originales
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Hybrid DotSimulate Diffusion - Fast & Quality</title>
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
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
            margin-bottom: 8px;
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
            letter-spacing: 1px;
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
            font-size: 18px;
            padding: 18px 35px;
        }
        
        .totem-btn:hover {
            box-shadow: 0 0 25px #ff6b00;
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
            min-height: 100px;
            max-height: 150px;
        }
        
        .range-value {
            color: #00ff88;
            font-weight: bold;
            text-align: center;
            font-size: 18px;
            margin-top: 8px;
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
            min-width: 300px;
            z-index: 1000;
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
        
        .status-indicator {
            position: absolute;
            top: 15px;
            left: 15px;
            width: 24px;
            height: 24px;
            border-radius: 50%;
            background: #ff4444;
            transition: all 0.3s;
        }
        
        .status-indicator.active {
            background: #00ff88;
            box-shadow: 0 0 20px #00ff88;
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
            max-width: 70vh;
            max-height: 90vh;
            width: auto;
            height: auto;
            border: none;
            border-radius: 0;
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
        
        .performance-mode {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .mode-label {
            font-size: 12px;
            color: #999;
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
            <div style="color: #00ff88; font-size: 20px; font-weight: bold;">HYBRID DOTSIMULATE MODE</div>
            <div style="margin-top: 10px;">Fast AI transformation with texture enhancement</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">Performance Metrics</div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Processing:</span>
            <span id="latency" class="stat-value">0</span>ms
        </div>
        <div class="stat-row">
            <span>Preprocessing:</span>
            <span id="preprocessing" class="stat-value">0</span>ms
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
            <span id="currentStrength" class="stat-value">0.5</span>
        </div>
        <div class="stat-row">
            <span>Guidance:</span>
            <span id="currentGuidance" class="stat-value">1.0</span>
        </div>
        <div class="stat-row">
            <span>Inference Steps:</span>
            <span id="inferenceSteps" class="stat-value">1</span>
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
            <div class="control-label">Custom Prompt</div>
            <textarea id="customPrompt" placeholder="Enter your prompt here... (e.g., cyberpunk style, anime character, oil painting)">cyberpunk style portrait</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Transformation Strength</div>
            <input type="range" id="strengthSlider" min="0.3" max="0.8" step="0.05" value="0.5" oninput="updateStrengthValue()">
            <div class="range-value" id="strengthValue">0.5</div>
            <div class="mode-label">Lower = Faster | Higher = More transformation</div>
        </div>
        
        <div class="control-group">
            <div class="control-label">Guidance Scale (Speed vs Quality)</div>
            <input type="range" id="guidanceSlider" min="1.0" max="5.0" step="0.5" value="1.0" oninput="updateGuidanceValue()">
            <div class="range-value" id="guidanceValue">1.0</div>
            <div class="mode-label">1.0 = Fastest | 5.0 = Best quality</div>
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
        
        let frameCount = 0;
        let totalFrames = 0;
        let lastTime = Date.now();
        let latencies = [];
        let preprocessingTimes = [];
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
        }
        
        function updateGuidanceValue() {
            const value = parseFloat(document.getElementById('guidanceSlider').value);
            document.getElementById('guidanceValue').textContent = value.toFixed(1);
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
                    console.log('Connected to Hybrid DotSimulate server');
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
            
            const prompt = document.getElementById('customPrompt').value || "professional photo";
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            const guidance = parseFloat(document.getElementById('guidanceSlider').value);
            
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.9);
            ws.send(JSON.stringify({
                image: imageData,
                prompt: prompt,
                strength: strength,
                guidance_scale: guidance,
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
            
            if (data.preprocessing_time) {
                preprocessingTimes.push(data.preprocessing_time);
                if (preprocessingTimes.length > 50) preprocessingTimes.shift();
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
                
                if (preprocessingTimes.length > 0) {
                    const avgPreprocessing = preprocessingTimes.reduce((a, b) => a + b) / preprocessingTimes.length;
                    document.getElementById('preprocessing').textContent = Math.round(avgPreprocessing);
                }
            }
            
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skipRate').textContent = data.stats.skip_rate.toFixed(1);
                document.getElementById('currentStrength').textContent = data.stats.strength;
                document.getElementById('currentGuidance').textContent = data.stats.guidance_scale;
                document.getElementById('inferenceSteps').textContent = data.stats.steps;
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
    print("🚀 Cliente conectado - Hybrid DotSimulate Mode")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if 'image' not in data:
                continue
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                if input_image.size != (512, 512):
                    input_image = input_image.resize((512, 512), Image.Resampling.LANCZOS)
                
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            prompt = data.get('prompt', 'professional photo')
            strength = data.get('strength', 0.5)
            guidance_scale = data.get('guidance_scale', 1.0)
            
            processor.add_frame(
                input_image, 
                prompt,
                strength,
                guidance_scale,
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
                    'preprocessing_time': result.get('preprocessing_time', 0),
                    'timestamp': result['timestamp'],
                    'enhanced_prompt': result.get('enhanced_prompt', ''),
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎨 HYBRID DOTSIMULATE DIFFUSION - FAST & QUALITY")
    print("="*80)
    print("⚡ LCM Dreamshaper v7 con 1-2 pasos adaptativos")
    print("🎨 Preprocessing de texturas optimizado")
    print("🔄 Temporal smoothing ligero")
    print("📝 Custom prompts con enhancement")
    print("📺 Modo Totem incluido")
    print("🎯 Objetivo: 15-20 FPS con calidad mejorada")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
