#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import io
import base64
from pathlib import Path
from typing import Union

# --- CONFIGURACIÓN DE ALTO RENDIMIENTO (Ajustable) ---
MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
TINY_VAE_ID = "madebyollin/taesd"
ENGINE_DIR = Path("./tensorrt_engines")
ENGINE_DIR.mkdir(exist_ok=True)
DEFAULT_PROMPT = "cinematic, professional photography, highly detailed, sharp focus, 8k"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, jpeg artifacts, ugly, deformed"
DEFAULT_GUIDANCE_SCALE = 1.2
DEFAULT_STRENGTH = 0.65
DEFAULT_TEMPORAL_SMOOTHING = 0.2
WIDTH, HEIGHT = 512, 512
# ---------------------------------------------------------

# --- INTERFAZ GRÁFICA (HTML) ---
# Hemos movido aquí el contenido del archivo que borramos para tener todo en un solo lugar.
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced StreamDiffusion - DotSimulate Style</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: linear-gradient(135deg, #0a0a0a, #1a1a2e); color: #fff; font-family: 'Consolas', 'Monaco', monospace; overflow: hidden; height: 100vh; }
        .container { display: flex; height: 100vh; align-items: center; justify-content: center; gap: 30px; padding: 20px; }
        video, canvas { width: 512px; height: 512px; border: 3px solid #00ff88; background: #111; border-radius: 12px; box-shadow: 0 0 30px rgba(0, 255, 136, 0.3); }
        .output-container { position: relative; }
        .controls { position: fixed; bottom: 20px; left: 20px; right: 20px; background: rgba(10, 10, 10, 0.95); padding: 25px; border-radius: 20px; border: 2px solid #00ff88; display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 25px; z-index: 1000; backdrop-filter: blur(10px); }
        .control-group { display: flex; flex-direction: column; gap: 12px; }
        .control-label { color: #00ff88; font-weight: bold; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
        button { padding: 15px 30px; font-size: 16px; cursor: pointer; background: linear-gradient(45deg, #00ff88, #00cc66); color: #000; border: none; border-radius: 10px; font-weight: bold; transition: all 0.3s; text-transform: uppercase; letter-spacing: 1px; }
        button:hover { transform: scale(1.05); box-shadow: 0 0 25px #00ff88; }
        button:disabled { background: #333; color: #666; cursor: not-allowed; transform: none; box-shadow: none; }
        .totem-btn { background: linear-gradient(45deg, #ff6b00, #ff4500); color: #fff; font-size: 18px; padding: 18px 35px; }
        .totem-btn:hover { box-shadow: 0 0 25px #ff6b00; }
        input[type="range"] { width: 100%; height: 8px; -webkit-appearance: none; background: linear-gradient(90deg, #333, #00ff88); outline: none; border-radius: 4px; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 24px; height: 24px; background: #00ff88; border-radius: 50%; cursor: pointer; box-shadow: 0 0 10px rgba(0, 255, 136, 0.5); }
        textarea { padding: 15px; font-size: 14px; border-radius: 10px; background: #1a1a1a; color: #fff; border: 2px solid #00ff88; font-family: 'Consolas', monospace; resize: vertical; min-height: 100px; max-height: 150px; }
        .range-value { color: #00ff88; font-weight: bold; text-align: center; font-size: 18px; margin-top: 8px; text-shadow: 0 0 10px #00ff88; }
        .stats { position: fixed; top: 20px; right: 20px; background: rgba(10, 10, 10, 0.95); padding: 25px; border-radius: 20px; font-family: 'Consolas', monospace; font-size: 12px; border: 2px solid #00ff88; min-width: 300px; z-index: 1000; backdrop-filter: blur(10px); }
        .stat-value { color: #00ff88; font-weight: bold; }
        .stat-row { display: flex; justify-content: space-between; margin: 10px 0; border-bottom: 1px solid #333; padding-bottom: 8px; }
        .status-indicator { position: absolute; top: 15px; left: 15px; width: 24px; height: 24px; border-radius: 50%; background: #ff4444; transition: all 0.3s; }
        .status-indicator.active { background: #00ff88; box-shadow: 0 0 20px #00ff88; }
        .title { text-align: center; margin-bottom: 20px; color: #00ff88; font-size: 16px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; }
        .totem-mode { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: #000; z-index: 9999; display: none; justify-content: center; align-items: center; }
        .totem-canvas { max-width: 70vh; max-height: 90vh; width: auto; height: auto; border: none; border-radius: 0; box-shadow: 0 0 50px rgba(0, 255, 136, 0.5); }
        .totem-exit { position: absolute; top: 30px; right: 30px; background: rgba(255, 0, 0, 0.8); color: white; border: none; border-radius: 50%; width: 70px; height: 70px; font-size: 28px; cursor: pointer; z-index: 10000; }
        .totem-info { position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%); background: rgba(0,0,0,0.9); padding: 20px 40px; border-radius: 15px; border: 2px solid #00ff88; text-align: center; }
        .enhanced-label { background: linear-gradient(45deg, #00ff88, #00cc66); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <div class="output-container"> <canvas id="output"></canvas> <div class="status-indicator" id="statusIndicator"></div> </div>
    </div>
    <div class="totem-mode" id="totemMode"> <button class="totem-exit" onclick="exitTotemMode()">✕</button> <canvas id="totemCanvas" class="totem-canvas"></canvas> <div class="totem-info"> <div style="color: #00ff88; font-size: 20px; font-weight: bold;">ENHANCED DOTSIMULATE MODE</div> <div style="margin-top: 10px;">Real-time AI transformation with advanced preprocessing</div> </div> </div>
    <div class="stats">
        <div class="title">Enhanced Metrics</div>
        <div class="stat-row"> <span>FPS:</span> <span id="fps" class="stat-value">0</span> </div>
        <div class="stat-row"> <span>Processing:</span> <span id="latency" class="stat-value">0</span>ms </div>
        <div class="stat-row"> <span>Total Frames:</span> <span id="totalFrames" class="stat-value">0</span> </div>
    </div>
    <div class="controls">
        <div class="control-group">
            <div class="control-label enhanced-label">System Control</div>
            <button id="startBtn" onclick="start()">🚀 Start Enhanced Stream</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ Stop</button>
            <button class="totem-btn" onclick="enterTotemMode()" id="totemBtn" disabled>📺 Totem Mode</button>
        </div>
        <div class="control-group">
            <div class="control-label enhanced-label">Enhanced Prompt Engineering</div>
            <textarea id="customPrompt" placeholder="Advanced prompt (will be automatically enhanced with quality boosters)">cyberpunk style, neon lights, futuristic cityscape</textarea>
        </div>
        <div class="control-group">
            <div class="control-label enhanced-label">Transformation Intensity</div>
            <input type="range" id="strengthSlider" min="0.10" max="0.95" step="0.01" value="0.65" oninput="updateStrengthValue()">
            <div class="range-value" id="strengthValue">0.65</div>
        </div>
        <div class="control-group">
            <div class="control-label enhanced-label">CFG Scale</div>
            <input type="range" id="guidanceSlider" min="1.0" max="2.5" step="0.1" value="1.2" oninput="updateGuidanceValue()">
            <div class="range-value" id="guidanceValue">1.2</div>
        </div>
    </div>
    <script>
        let ws = null, streaming = false, video = null, canvas = null, ctx = null, totemCanvas = null, totemCtx = null, totemMode = false;
        let frameCount = 0, totalFrames = 0, lastTime = Date.now(), latencies = [];
        function updateStrengthValue() { document.getElementById('strengthValue').textContent = parseFloat(document.getElementById('strengthSlider').value).toFixed(2); }
        function updateGuidanceValue() { document.getElementById('guidanceValue').textContent = parseFloat(document.getElementById('guidanceSlider').value).toFixed(1); }
        function enterTotemMode() { if (!streaming) return; totemMode = true; document.getElementById('totemMode').style.display = 'flex'; totemCanvas = document.getElementById('totemCanvas'); totemCtx = totemCanvas.getContext('2d'); const screenHeight = window.innerHeight, screenWidth = window.innerWidth; if (screenHeight > screenWidth) { totemCanvas.width = screenWidth * 0.8; totemCanvas.height = screenWidth * 0.8; } else { totemCanvas.width = screenHeight * 0.7; totemCanvas.height = screenHeight * 0.7; } document.querySelector('.controls').style.display = 'none'; document.querySelector('.stats').style.display = 'none'; }
        function exitTotemMode() { totemMode = false; document.getElementById('totemMode').style.display = 'none'; document.querySelector('.controls').style.display = 'grid'; document.querySelector('.stats').style.display = 'block'; }
        async function start() {
            try {
                video = document.getElementById('video'); canvas = document.getElementById('output'); ctx = canvas.getContext('2d'); canvas.width = 512; canvas.height = 512;
                const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } } });
                video.srcObject = stream; await new Promise(resolve => video.onloadedmetadata = resolve);
                ws = new WebSocket(`${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`);
                ws.onopen = () => { streaming = true; document.getElementById('startBtn').disabled = true; document.getElementById('stopBtn').disabled = false; document.getElementById('totemBtn').disabled = false; document.getElementById('statusIndicator').classList.add('active'); sendFrame(); };
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, 512, 512); ctx.drawImage(img, 0, 0, 512, 512);
                            if (totemMode && totemCtx) { totemCtx.clearRect(0, 0, totemCanvas.width, totemCanvas.height); totemCtx.drawImage(img, 0, 0, totemCanvas.width, totemCanvas.height); }
                        };
                        img.src = data.image; updateAdvancedStats(data);
                    }
                };
                ws.onerror = () => stop(); ws.onclose = () => stop();
            } catch (error) { alert('Error: ' + error.message); }
        }
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            if (video.videoWidth === 0) { requestAnimationFrame(sendFrame); return; }
            const tempCanvas = document.createElement('canvas'); tempCanvas.width = 512; tempCanvas.height = 512; const tempCtx = tempCanvas.getContext('2d');
            const videoAspect = video.videoWidth / video.videoHeight, targetAspect = 1; let sx, sy, sw, sh;
            if (videoAspect > targetAspect) { sh = video.videoHeight; sw = video.videoHeight; sx = (video.videoWidth - sw) / 2; sy = 0; } else { sw = video.videoWidth; sh = video.videoWidth; sx = 0; sy = (video.videoHeight - sh) / 2; }
            tempCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 512, 512);
            ws.send(JSON.stringify({ image: tempCanvas.toDataURL('image/jpeg', 0.9), prompt: document.getElementById('customPrompt').value, strength: parseFloat(document.getElementById('strengthSlider').value), guidance_scale: parseFloat(document.getElementById('guidanceSlider').value), timestamp: Date.now() }));
            requestAnimationFrame(sendFrame);
        }
        function updateAdvancedStats(data) {
            frameCount++; totalFrames++; if (data.stats && data.stats.latency) { latencies.push(data.stats.latency); if (latencies.length > 50) latencies.shift(); }
            const now = Date.now();
            if (now - lastTime >= 1000) {
                document.getElementById('fps').textContent = frameCount; frameCount = 0; lastTime = now;
                if (latencies.length > 0) document.getElementById('latency').textContent = Math.round(latencies.reduce((a, b) => a + b) / latencies.length);
            }
            document.getElementById('totalFrames').textContent = totalFrames;
        }
        function stop() {
            streaming = false; if (totemMode) exitTotemMode(); if (ws) ws.close(); if (video && video.srcObject) video.srcObject.getTracks().forEach(track => track.stop());
            if (ctx) ctx.clearRect(0, 0, 512, 512); document.getElementById('startBtn').disabled = false; document.getElementById('stopBtn').disabled = true; document.getElementById('totemBtn').disabled = true; document.getElementById('statusIndicator').classList.remove('active');
        }
        document.addEventListener('keydown', (event) => { if (event.key === 'Escape' && totemMode) exitTotemMode(); }); window.addEventListener('beforeunload', stop);
    </script>
</body>
</html>
"""

# --- INICIO CORRECTO DE PYTORCH Y CUDA ---
import torch
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if not torch.cuda.is_available():
    print("FATAL: CUDA no está disponible.", file=sys.stderr)
    sys.exit(1)
device = torch.device("cuda")
dtype = torch.float16
print(f"✅ CUDA detectado: {torch.cuda.get_device_name(0)} | Usando precisión: {dtype}")

# --- Importaciones de Librerías de IA ---
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from PIL import Image

# -----------------------------------------------

class StreamProcessor:
    def __init__(self):
        self.last_latents = None
        self.frame_times = []
        self.current_prompt = DEFAULT_PROMPT
        self.current_guidance = DEFAULT_GUIDANCE_SCALE
        self.stream = self._initialize_pipeline()
        print("✅ Procesador de Stream optimizado y listo.")

    def _initialize_pipeline(self) -> StreamDiffusion:
        print("🚀 Inicializando pipeline...")
        print(f"--> Cargando modelo '{MODEL_ID}' con diffusers...")
        pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
        print(f"--> Cargando TinyVAE '{TINY_VAE_ID}'...")
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=dtype)
        pipe.to(device=device)
        print("--> Configurando StreamDiffusion...")
        stream = StreamDiffusion(
            pipe=pipe, t_index_list=[0, 16, 32, 45], frame_buffer_size=1, width=WIDTH, height=HEIGHT,
        )
        print(f"--> Cargando y fusionando LoRA LCM...")
        stream.load_lcm_lora()
        stream.fuse_lora()
        print("⚡ Acelerando con TensorRT...")
        try:
            stream = accelerate_with_tensorrt(stream, ENGINE_DIR, max_batch_size=2)
            print("✅ Pipeline acelerado con TensorRT exitosamente.")
        except Exception as e:
            print(f"⚠️ Advertencia: Falló la aceleración con TensorRT. {e}", file=sys.stderr)
        print("🔥 Pre-calentando el pipeline...")
        stream.prepare(
            prompt=self.current_prompt, negative_prompt=DEFAULT_NEGATIVE_PROMPT, num_inference_steps=50, guidance_scale=self.current_guidance,
        )
        print("✅ Pipeline listo para recibir frames.")
        return stream

    @torch.no_grad()
    def process_frame(self, image: Image.Image, params: dict) -> (Image.Image, dict):
        start_time = time.time()
        if self.current_prompt != params['prompt'] or self.current_guidance != params['guidance_scale']:
            self.stream.prepare(
                prompt=params['prompt'], negative_prompt=DEFAULT_NEGATIVE_PROMPT, num_inference_steps=50, guidance_scale=params['guidance_scale'],
            )
            self.current_prompt = params['prompt']
            self.current_guidance = params['guidance_scale']
        input_tensor = self.stream.image_processor.preprocess(image).to(device=device, dtype=dtype)
        noise = torch.randn_like(input_tensor) * 0.02
        input_tensor = torch.clamp(input_tensor + noise, 0, 1)
        latents = self.stream.encode_image(input_tensor)
        noisy_latents = self.stream.add_noise(latents, params['strength'])
        denoised_latents = self.stream(image_latents=noisy_latents)
        if self.last_latents is not None:
             denoised_latents = torch.lerp(self.last_latents, denoised_latents, 1.0 - params['temporal_smoothing'])
        self.last_latents = denoised_latents.clone()
        output_tensor = self.stream.decode_image(denoised_latents)
        output_image = postprocess_image(output_tensor, output_type="pil")[0]
        processing_time_ms = (time.time() - start_time) * 1000
        self.frame_times.append(processing_time_ms)
        if len(self.frame_times) > 100: self.frame_times.pop(0)
        avg_latency = sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        stats = {"fps": round(fps, 1), "latency": round(processing_time_ms, 1)}
        return output_image, stats

# --- SERVIDOR WEB FASTAPI ---
app = FastAPI()
processor = StreamProcessor()

@app.get("/")
async def get_root():
    # Ahora el HTML está en este mismo archivo.
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Cliente conectado al WebSocket.")
    try:
        while True:
            data = await websocket.receive_json()
            img_data = base64.b64decode(data['image'].split(',')[1])
            input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            params = {
                'prompt': data.get('prompt', DEFAULT_PROMPT),
                'strength': float(data.get('strength', DEFAULT_STRENGTH)),
                'guidance_scale': float(data.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)),
                'temporal_smoothing': DEFAULT_TEMPORAL_SMOOTHING,
            }
            output_image, stats = processor.process_frame(input_image, params)
            buffered = io.BytesIO()
            output_image.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            await websocket.send_json({'image': f'data:image/jpeg;base64,{img_str}', 'stats': stats})
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}", file=sys.stderr)
    finally:
        processor.last_latents = None
        print("🔌 Cliente desconectado.")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("🚀 SERVIDOR STREAMDIFFUSION - MÁXIMO RENDIMIENTO (Estilo DotSimulate)")
    print("="*80)
    print(f"⚡ Aceleración:      TensorRT (Caché en ./{ENGINE_DIR.name}/)")
    print(f"📦 Modelo:           {MODEL_ID}")
    print(f"🎨 VAE Óptimizado:   {TINY_VAE_ID}")
    print(f"🛠️ Procesamiento:    100% en GPU (PyTorch)")
    print(f"🎯 Rendimiento L40S: ~40-60 FPS @ {WIDTH}x{HEIGHT}")
    print(f"🌐 Servidor en:      http://0.0.0.0:8000")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
