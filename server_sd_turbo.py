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
from diffusers import AutoPipelineForImage2Image
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

class SDTurboDiffusion:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.resolution = 512  # SD-Turbo funciona mejor a 512
        
        # Threading simple
        self.current_task = None
        self.last_result = None
        self.processing = False
        
        # Counters
        self.total_frames = 0
        self.processed_frames = 0
        
        self.init_model()
        
    def init_model(self):
        print("\n🚀 Inicializando SD-Turbo...")
        
        # SD-Turbo
        model_id = "stabilityai/sd-turbo"
        
        print(f"📥 Cargando {model_id}...")
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            variant="fp16",
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # No scheduler changes needed for SD-Turbo
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
        
        # Warmup
        print("🔥 Calentando pipeline...")
        try:
            dummy = Image.new('RGB', (self.resolution, self.resolution), (128, 128, 128))
            with torch.no_grad():
                _ = self.pipe(
                    prompt="test",
                    image=dummy,
                    num_inference_steps=1,
                    strength=0.5,
                    guidance_scale=0.0  # SD-Turbo usa 0.0
                ).images[0]
            torch.cuda.empty_cache()
            print("✅ Pipeline listo!")
        except Exception as e:
            print(f"⚠️  Warmup error: {e}")
        
        # Start processing
        self.processing = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("✅ Processing thread iniciado\n")
    
    def _process_loop(self):
        """Loop de procesamiento para SD-Turbo"""
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
                
                # Generate with SD-Turbo settings
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        result = self.pipe(
                            prompt=task.get('prompt', '') or '',
                            image=image,
                            num_inference_steps=1,  # SD-Turbo siempre usa 1
                            strength=task.get('strength', 0.5),
                            guidance_scale=0.0  # SD-Turbo siempre usa 0.0
                        ).images[0]
                
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
processor = SDTurboDiffusion()

# HTML con WebSocket corregido
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SD-Turbo A6000</title>
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
        
        .model-info {
            color: #ff6b00;
            font-size: 14px;
            margin-top: 10px;
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
            <div style="color: #00ff88; font-size: 20px; font-weight: bold;">SD-TURBO MODE</div>
            <div style="margin-top: 10px;">Stable Diffusion Turbo - Ultra Fast</div>
        </div>
    </div>
    
    <div class="stats">
        <div class="title">SD-Turbo Performance</div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Processing:</span>
            <span id="latency" class="stat-value">0</span>ms
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
        <div class="model-info">Model: SD-Turbo 512x512</div>
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
            <textarea id="customPrompt" placeholder="Enter your prompt (SD-Turbo works best with simple prompts)">cyberpunk portrait</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Strength: <span id="strengthValue">0.5</span></div>
            <input type="range" id="strengthSlider" min="0.3" max="0.8" step="0.05" value="0.5" oninput="updateStrengthValue()">
            <div style="color: #999; font-size: 11px; margin-top: 5px;">SD-Turbo optimal: 0.5-0.7</div>
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
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
            document.getElementById('currentStrength').textContent = value.toFixed(2);
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
                
                // AQUÍ ESTÁ LA CORRECCIÓN DEL WEBSOCKET
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                console.log('Connecting to:', wsUrl);
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('Connected to SD-Turbo server');
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
            
            if (data.time) {
                latencies.push(data.time);
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
                document.getElementById('totalFrames').textContent = data.stats.total;
                document.getElementById('skipRate').textContent = data.stats.skip_rate;
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
    print("🚀 Cliente conectado - SD-Turbo Mode")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
            except Exception as e:
                print(f"❌ Error decodificando imagen: {e}")
                continue
            
            prompt = data.get('prompt', 'portrait')
            strength = data.get('strength', 0.5)
            
            processor.add_frame(
                input_image, 
                prompt,
                strength
            )
            
            result = processor.get_result()
            
            if result:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'time': result['time'],
                    'stats': result['stats']
                })
            
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}")
    finally:
        print("🔌 Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🎨 SD-TURBO A6000 OPTIMIZED")
    print("="*80)
    print("⚡ Stable Diffusion Turbo - 1 step inference")
    print("🎯 Resolution: 512x512 (native)")
    print("🔧 Guidance Scale: 0.0 (SD-Turbo optimized)")
    print("📺 Custom Prompts + Totem Mode")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
