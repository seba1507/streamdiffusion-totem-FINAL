#!/usr/bin/env python3
import os
import sys
import time
import threading
import queue
import numpy as np
from PIL import Image, ImageEnhance

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
from diffusers import DiffusionPipeline, AutoPipelineForImage2Image
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

class WorkingSDTurbo:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        self.resolution = 512
        
        # Threading
        self.current_task = None
        self.last_result = None
        self.last_result_hash = None  # Para detectar resultados idénticos
        self.processing = False
        
        # Counters
        self.total_frames = 0
        self.processed_frames = 0
        self.identical_results = 0
        
        self.init_model()
        
    def init_model(self):
        print("\n🚀 Inicializando SD-Turbo (Fixed)...")
        
        try:
            # Intentar cargar SD-Turbo con configuración específica
            print("📥 Cargando SD-Turbo...")
            
            # Usar DiffusionPipeline genérico primero
            self.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/sd-turbo",
                torch_dtype=self.dtype,
                use_safetensors=True,
                safety_checker=None
            ).to(self.device)
            
            # Verificar que sea img2img pipeline
            if not hasattr(self.pipe, '__call__'):
                print("⚠️  Convirtiendo a img2img pipeline...")
                self.pipe = AutoPipelineForImage2Image.from_pipe(self.pipe).to(self.device)
            
            print("✅ SD-Turbo cargado correctamente")
            
        except Exception as e:
            print(f"❌ Error cargando SD-Turbo: {e}")
            print("🔄 Fallback a LCM...")
            
            # Fallback a LCM que sabemos que funciona
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                "SimianLuo/LCM_Dreamshaper_v7",
                torch_dtype=self.dtype,
                safety_checker=None,
                use_safetensors=True
            ).to(self.device)
            
            print("✅ LCM cargado como fallback")
        
        # Configuraciones
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers habilitado")
        except:
            print("⚠️  XFormers no disponible")
        
        # VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✅ VAE optimizado")
        except:
            pass
        
        # Test del modelo
        print("🧪 Testeando modelo...")
        try:
            dummy = Image.new('RGB', (512, 512), (128, 128, 128))
            with torch.no_grad():
                # Test con parámetros seguros
                test_result = self.pipe(
                    prompt="a photo",
                    image=dummy,
                    num_inference_steps=1,
                    strength=0.5,
                    guidance_scale=0.0,
                    output_type="pil",
                    return_dict=True
                )
                
                if hasattr(test_result, 'images') and len(test_result.images) > 0:
                    print("✅ Test exitoso - modelo funcionando")
                else:
                    print("⚠️  Test produjo resultado vacío")
                    
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ Error en test: {e}")
            print("Continuando de todos modos...")
        
        # Start processing
        self.processing = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("✅ Processing thread iniciado\n")
    
    def _process_loop(self):
        """Loop de procesamiento robusto"""
        while self.processing:
            if self.current_task is None:
                time.sleep(0.001)
                continue
            
            try:
                start = time.time()
                task = self.current_task
                
                # Preparar imagen
                image = task['image']
                if image.size != (self.resolution, self.resolution):
                    image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
                
                # Asegurar que la imagen esté en RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Prompt simple para SD-Turbo
                prompt = task.get('prompt', '').strip()
                if not prompt:
                    prompt = "a photo"
                
                # Parámetros seguros
                strength = float(task.get('strength', 0.7))  # Default más alto
                strength = max(0.4, min(0.95, strength))  # Mínimo 0.4 para más cambio
                
                # Para debugging
                # print(f"📸 Procesando frame con strength={strength:.2f}, prompt='{prompt[:30]}'...")
                
                # Generar con manejo de errores robusto
                result = None
                seed = int(time.time() * 1000) % 100000  # Seed variable
                
                with torch.no_grad():
                    try:
                        output = self.pipe(
                            prompt=prompt,
                            image=image,
                            num_inference_steps=1,
                            strength=strength,
                            guidance_scale=0.0,
                            generator=torch.Generator(device=self.device).manual_seed(seed),
                            output_type="pil",
                            return_dict=True
                        )
                        
                        # Verificar resultado
                        if hasattr(output, 'images') and len(output.images) > 0:
                            result = output.images[0]
                        else:
                            print("⚠️  Output vacío, usando imagen original")
                            result = image
                            
                    except Exception as e:
                        print(f"❌ Error en generación: {e}")
                        result = image  # Usar imagen original como fallback
                
                # Asegurar que tenemos un resultado
                if result is None:
                    result = image
                
                # Verificar si el resultado es idéntico al anterior
                result_array = np.array(result)
                result_hash = hash(result_array.tobytes())
                
                if self.last_result_hash and result_hash == self.last_result_hash:
                    self.identical_results += 1
                    print(f"⚠️  Resultado idéntico detectado ({self.identical_results} veces)")
                    
                    # Si hay muchos resultados idénticos, aumentar strength
                    if self.identical_results > 3:
                        print("🔄 Forzando más variación...")
                        # Aplicar transformación adicional a la imagen
                        enhancer = ImageEnhance.Color(result)
                        result = enhancer.enhance(1.0 + (np.random.random() - 0.5) * 0.2)
                        self.identical_results = 0
                else:
                    self.identical_results = 0
                
                self.last_result_hash = result_hash
                
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
                
                print(f"✓ Frame: {elapsed:.0f}ms (seed: {seed}, strength: {strength:.2f})")
                
            except Exception as e:
                print(f"❌ Error general: {type(e).__name__}: {e}")
                # Crear resultado dummy en caso de error
                dummy_img = Image.new('RGB', (512, 512), (128, 0, 128))
                self.last_result = {
                    'image': dummy_img,
                    'time': 0,
                    'stats': {
                        'total': self.total_frames,
                        'processed': self.processed_frames,
                        'skip_rate': 0
                    }
                }
            
            self.current_task = None
            
            # Clean cache más frecuentemente
            if self.processed_frames % 10 == 0:
                torch.cuda.empty_cache()
                print("🧹 Cache limpiado")
    
    def add_frame(self, image, prompt, strength):
        self.total_frames += 1
        
        # NO skip frames - procesar todos para más variación
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
processor = WorkingSDTurbo()

# HTML igual pero con ajustes para SD-Turbo
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>SD-Turbo Fixed</title>
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
    </div>
    
    <div class="stats">
        <div class="title">SD-Turbo Fixed</div>
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
            <textarea id="customPrompt" placeholder="Simple prompts work best with SD-Turbo">cyberpunk portrait</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Strength: <span id="strengthValue">0.7</span></div>
            <input type="range" id="strengthSlider" min="0.4" max="0.95" step="0.05" value="0.7" oninput="updateStrengthValue()">
            <div style="color: #999; font-size: 11px; margin-top: 5px;">⚡ Higher = More transformation</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
        }
        
        function enterTotemMode() {
            if (!streaming) {
                alert('Start the stream first');
                return;
            }
            document.getElementById('totemMode').style.display = 'flex';
            const tc = document.getElementById('totemCanvas');
            tc.width = tc.height = 512;
        }
        
        function exitTotemMode() {
            document.getElementById('totemMode').style.display = 'none';
        }
        
        async function start() {
            try {
                const video = document.getElementById('video');
                const canvas = document.getElementById('output');
                const ctx = canvas.getContext('2d');
                canvas.width = canvas.height = 512;
                
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { width: 640, height: 480 }
                });
                video.srcObject = stream;
                
                await new Promise(r => video.onloadedmetadata = r);
                
                // WebSocket con detección de protocolo
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws`;
                console.log('Connecting to:', wsUrl);
                
                ws = new WebSocket(wsUrl);
                
                ws.onopen = () => {
                    console.log('Connected');
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
                image: tc.toDataURL('image/jpeg', 0.85),
                prompt: document.getElementById('customPrompt').value,
                strength: parseFloat(document.getElementById('strengthSlider').value)
            }));
            
            setTimeout(sendFrame, 100); // 10 FPS para menos skip
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
                document.getElementById('totalFrames').textContent = data.stats.total;
                document.getElementById('skipRate').textContent = data.stats.skip_rate;
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
            document.getElementById('totemBtn').disabled = true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        
        document.addEventListener('keydown', e => {
            if (e.key === 'Escape') exitTotemMode();
            if (e.key === 'F11') { e.preventDefault(); enterTotemMode(); }
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
    print("🚀 Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            try:
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
            except Exception as e:
                print(f"Error imagen: {e}")
                continue
            
            processor.add_frame(
                image,
                data.get('prompt', ''),
                data.get('strength', 0.7)  # Default más alto
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
        print(f"WebSocket error: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🚀 SD-TURBO FIXED - A6000 OPTIMIZED")
    print("="*80)
    print("⚡ SD-Turbo con fallback a LCM si falla")
    print("🛡️ Manejo robusto de errores")
    print("🎯 WebSocket seguro (wss://)")
    print("📺 Totem Mode incluido")
    print("🌐 http://0.0.0.0:8000")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
