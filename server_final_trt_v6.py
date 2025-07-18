#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor StreamDiffusion TOTEM – 100 % TensorRT (VAE‑Encoder + UNet + VAE‑Decoder)

► Requisitos previos
    tools/tensorrt_engines_img2img/
        ├─ vae_encoder.engine   (input: img 1×3×512×512)
        ├─ unet.engine          (inputs: latent 1×4×64×64, timestep 1, enc_hidden 1×77×768)
        └─ vae_decoder.engine   (input: latent 1×4×64×64)

► Ejecuta:
    python server_final_trt.py
"""

# ----------- IMPORTS BÁSICOS --------------------------------------------------
import os, sys, time, io, base64, asyncio
from pathlib import Path
from typing import Tuple

import polygraphy_patch          # parche compatibilidad TensorRT 10
import randn_patch               # parche randn multi‑generator

# ----------- CONFIG -----------------------------------------------------------
MODEL_ID          = "SimianLuo/LCM_Dreamshaper_v7"
TINY_VAE_ID       = "madebyollin/taesd"
ENGINE_DIR        = Path("tools/tensorrt_engines_img2img")   # <- TU carpeta engines
DEFAULT_PROMPT    = "cinematic, professional photography, highly detailed, sharp focus, 8k"
DEFAULT_NEGATIVE  = "blurry, low quality, jpeg artifacts, ugly, deformed"
DEFAULT_GUIDANCE  = 1.2
DEFAULT_STRENGTH  = 0.65
WIDTH, HEIGHT     = 512, 512


# --- INTERFAZ GRÁFICA (HTML) ---
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



# ----------- CUDA / PYTORCH ---------------------------------------------------
import torch
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
assert torch.cuda.is_available(), "CUDA no disponible"
device, dtype = torch.device("cuda"), torch.float16
print(f"✅ CUDA: {torch.cuda.get_device_name(0)} | dtype={dtype}")

# ----------- IA LIBS ----------------------------------------------------------
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from PIL import Image

# ----------- WRAPPER TENSORRT -------------------------------------------------
import tensorrt as trt
class TrtEngine:
    """
    Wrapper minimalista compat‑TensorRT 8/9 (num_bindings) y 10 (num_io_tensors).
    Soporta N inputs, 1 output – suficiente para VAE Encoder/Decoder y UNet.
    """
    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        # --- compatibilidad versiones ---
        if hasattr(self.engine, "num_bindings"):
            self.n_bindings = self.engine.num_bindings
        else:                             # TensorRT 10
            self.n_bindings = self.engine.num_io_tensors
        # ---------------------------------
        self.bindings = [None] * self.n_bindings
        self.out      = None
        self.stream   = torch.cuda.current_stream().cuda_stream
        print(f"📦 TRT cargado: {Path(path).name}")

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == self.n_bindings - 1, \
            f"Se esperaban {self.n_bindings-1} tensores, recibidos {len(inputs)}"
        for idx, t in enumerate(inputs):
            self.bindings[idx] = int(t.data_ptr())
        if self.out is None:
            # índice del output = último binding
            if hasattr(self.engine, "get_binding_shape"):
                out_shape = tuple(self.engine.get_binding_shape(self.n_bindings - 1))
            else:  # fallback polygraphy_patch ya añadió compatibilidad
                tensor_name = self.engine.get_tensor_name(self.n_bindings - 1)
                out_shape   = tuple(self.engine.get_tensor_shape(tensor_name))
            self.out = torch.empty(out_shape, dtype=inputs[0].dtype, device=inputs[0].device)
            self.bindings[-1] = int(self.out.data_ptr())
        if hasattr(self.context, "execute_async_v2"):           # TRT 8/9
            self.context.execute_async_v2(self.bindings, self.stream)
        elif hasattr(self.context, "execute_v2"):               # ✅ TRT 10
            self.context.execute_v2(self.bindings)              # síncrono
        else:
            raise RuntimeError("Método de ejecución TensorRT no soportado")
        return self.out


# ----------- PIPELINE / PROCESSOR --------------------------------------------
class StreamProcessor:
    def __init__(self):
        self.frame_times = []
        self.current_prompt   = DEFAULT_PROMPT
        self.current_guidance = DEFAULT_GUIDANCE
        self.stream = self._init_pipeline()
        print("✅ Procesador listo (TensorRT full stack)")

    # ---- inicializar diffusers ------------------------------------------------
    def _init_pipeline(self) -> StreamDiffusion:
        print("🚀 Cargando modelo y configurando StreamDiffusion …")
        pipe = AutoPipelineForImage2Image.from_pretrained(MODEL_ID, torch_dtype=dtype)
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=dtype)
        pipe.to(device)

        sd = StreamDiffusion(
            pipe=pipe,
            t_index_list=[0],
            frame_buffer_size=1,
            width=WIDTH, height=HEIGHT,
        )
        sd.load_lcm_lora(); sd.fuse_lora()

        # ---- cargar engines TensorRT -----------------------------------------
        enc_path = ENGINE_DIR / "vae_encoder.engine"
        dec_path = ENGINE_DIR / "vae_decoder.engine"
        unet_path= ENGINE_DIR / "unet.engine"
        sd.trt_enc  = TrtEngine(str(enc_path))
        sd.trt_dec  = TrtEngine(str(dec_path))
        sd.trt_unet = TrtEngine(str(unet_path))

        # ---- monkey‑patch métodos --------------------------------------------
        def encode_image_trt(self, pil_img: Image.Image):
            img = self.image_processor.preprocess(pil_img, self.height, self.width).to(device, dtype)
            latent = self.trt_enc(img) * (1/0.18215)
            return latent
        sd.encode_image = encode_image_trt.__get__(sd)

        def decode_latent_trt(self, latent: torch.Tensor):
            return self.trt_dec(latent * 0.18215)
        sd.decode_latent = decode_latent_trt.__get__(sd)

        def predict_noise_trt(self, latent_in, t, enc_hidden):
            return self.trt_unet(latent_in, enc_hidden, t)
        sd._predict_noise = predict_noise_trt.__get__(sd)  # nombre según versión

        sd.prepare(prompt=self.current_prompt,
                   negative_prompt=DEFAULT_NEGATIVE,
                   num_inference_steps=50,
                   guidance_scale=self.current_guidance)
        return sd

    # ---- procesar un frame ----------------------------------------------------
    @torch.no_grad()
    def process_frame(self, img: Image.Image, params: dict) -> Tuple[Image.Image, dict]:
        start = time.time()

        if params["prompt"] != self.current_prompt or params["guidance_scale"] != self.current_guidance:
            self.stream.prepare(prompt=params["prompt"],
                                negative_prompt=DEFAULT_NEGATIVE,
                                num_inference_steps=50,
                                guidance_scale=params["guidance_scale"])
            self.current_prompt   = params["prompt"]
            self.current_guidance = params["guidance_scale"]

        if img.size != (512, 512):
            img = img.resize((512, 512), Image.LANCZOS)

        latents = self.stream(img)             # --- todo corre en TensorRT ---
        out_img = postprocess_image(latents, output_type="pil")[0]

        t_ms = (time.time() - start) * 1000
        self.frame_times.append(t_ms); self.frame_times = self.frame_times[-100:]
        avg = sum(self.frame_times)/len(self.frame_times)
        stats = {"fps": round(1000/avg,1), "latency": round(t_ms,1)}
        return out_img, stats

# ----------- FASTAPI / WEBSOCKET ---------------------------------------------
app = FastAPI()
processor = StreamProcessor()

@app.get("/")
async def root():
    return HTMLResponse(content=HTML_CONTENT, status_code=200)


@app.websocket("/ws")
async def ws_gateway(ws: WebSocket):
    await ws.accept(); print("✅ WebSocket conectado")
    try:
        while True:
            data = await ws.receive_json()
            img_bytes = base64.b64decode(data["image"].split(",")[1])
            frame_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            params = {
                "prompt": data.get("prompt", DEFAULT_PROMPT),
                "strength": float(data.get("strength", DEFAULT_STRENGTH)),
                "guidance_scale": float(data.get("guidance_scale", DEFAULT_GUIDANCE)),
            }
            out_img, stats = processor.process_frame(frame_img, params)
            buf = io.BytesIO(); out_img.save(buf, format="JPEG", quality=90)
            payload = base64.b64encode(buf.getvalue()).decode()
            await ws.send_json({"image": f"data:image/jpeg;base64,{payload}", "stats": stats})
    except Exception as e:
        print("❌ WebSocket error:", e)
    finally:
        print("🔌 WebSocket cerrado")

# ----------- MAIN -------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, textwrap
    print(textwrap.dedent(f"""
        ############################################################
        𝙎𝙩𝙧𝙚𝙖𝙢𝘿𝙞𝙛𝙛𝙪𝙨𝙞𝙤𝙣 TOTEM  –  TensorRT img2img  (512×512)
        ⚡ Engines  : {ENGINE_DIR}
        📦 Modelo  : {MODEL_ID}
        🎯 GPU     : {torch.cuda.get_device_name(0)}
        ############################################################
    """))
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
