#!/usr/bin/env python3
import os
import sys
import polygraphy_patch

import torch
import time
import asyncio
import io
import base64
from pathlib import Path
from PIL import Image
import numpy as np

# Configuración
MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
TINY_VAE_ID = "madebyollin/taesd"
ENGINE_DIR = Path("./tensorrt_engines")
ENGINE_DIR.mkdir(exist_ok=True)
WIDTH, HEIGHT = 512, 512

# HTML (simplificado para debugging)
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>StreamDiffusion Test</title>
    <style>
        body { background: #1a1a1a; color: white; font-family: monospace; }
        video, canvas { width: 512px; height: 512px; border: 2px solid #00ff88; }
        button { padding: 10px 20px; margin: 10px; }
        #status { color: #00ff88; margin: 10px; }
    </style>
</head>
<body>
    <h1>StreamDiffusion Test</h1>
    <div id="status">Disconnected</div>
    <video id="video" autoplay muted playsinline></video>
    <canvas id="output"></canvas>
    <br>
    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>
    <script>
        let ws = null, streaming = false;
        
        async function start() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('output');
            const ctx = canvas.getContext('2d');
            canvas.width = 512; canvas.height = 512;
            
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            video.srcObject = stream;
            
            ws = new WebSocket('ws://localhost:8000/ws');
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected';
                streaming = true;
                sendFrame();
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.image) {
                    const img = new Image();
                    img.onload = () => ctx.drawImage(img, 0, 0, 512, 512);
                    img.src = data.image;
                }
                if (data.fps) {
                    document.getElementById('status').textContent = `FPS: ${data.fps}`;
                }
            };
            
            ws.onerror = ws.onclose = () => {
                document.getElementById('status').textContent = 'Disconnected';
                stop();
            };
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            const video = document.getElementById('video');
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512; tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(video, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.9)
            }));
            
            requestAnimationFrame(sendFrame);
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            const video = document.getElementById('video');
            if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        }
    </script>
</body>
</html>
"""

# Configurar CUDA
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
device = torch.device("cuda")
dtype = torch.float16
print(f"✅ CUDA: {torch.cuda.get_device_name(0)}")

from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

class SimpleStreamProcessor:
    def __init__(self):
        self.stream = self._setup_stream()
        self.frame_count = 0
        self.start_time = time.time()
        
    def _setup_stream(self):
        print("🚀 Inicializando StreamDiffusion...")
        
        # Cargar pipeline
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, 
            torch_dtype=dtype,
            use_safetensors=True
        )
        
        # TinyVAE para velocidad
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=dtype)
        pipe = pipe.to(device)
        
        # StreamDiffusion con configuración básica
        stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=[32, 45],  # Solo 2 pasos para máxima velocidad
            torch_dtype=dtype,
            width=WIDTH,
            height=HEIGHT,
            do_add_noise=True,
            frame_buffer_size=1,
            use_denoising_batch=True,
            cfg_type="none",  # Sin CFG para máxima velocidad
        )
        
        # Cargar LCM-LoRA
        stream.load_lcm_lora()
        stream.fuse_lora()
        
        # TensorRT
        try:
            stream = accelerate_with_tensorrt(stream, ENGINE_DIR, max_batch_size=1)
            print("✅ TensorRT acelerado")
        except Exception as e:
            print(f"⚠️ TensorRT falló: {e}")
        
        # Preparar
        stream.prepare(
            prompt="beautiful scenery, high quality",
            num_inference_steps=50,
            guidance_scale=1.0,
        )
        
        return stream
    
    @torch.no_grad()
    def process(self, pil_image):
        # Asegurar tamaño correcto
        if pil_image.size != (WIDTH, HEIGHT):
            pil_image = pil_image.resize((WIDTH, HEIGHT), Image.LANCZOS)
        
        # Convertir a tensor
        image_tensor = self.stream.image_processor.preprocess(
            pil_image, 
            height=HEIGHT, 
            width=WIDTH
        ).to(device=device, dtype=dtype)
        
        # Procesar con StreamDiffusion (método simple)
        for _ in range(self.stream.batch_size - 1):
            self.stream(image=image_tensor)
        
        output_image = self.stream(image=image_tensor)
        
        # Postprocesar
        output_image = postprocess_image(output_image, output_type="pil")[0]
        
        # Calcular FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return output_image, round(fps, 1)

# FastAPI
app = FastAPI()
processor = SimpleStreamProcessor()

@app.get("/")
async def root():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Cliente conectado")
    
    try:
        while True:
            data = await websocket.receive_json()
            
            # Decodificar imagen
            img_data = base64.b64decode(data['image'].split(',')[1])
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Procesar
            output_image, fps = processor.process(pil_image)
            
            # Codificar resultado
            buffered = io.BytesIO()
            output_image.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Enviar
            await websocket.send_json({
                'image': f'data:image/jpeg;base64,{img_str}',
                'fps': fps
            })
            
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🔌 Desconectado")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("STREAMDIFFUSION SIMPLE - SIN ERRORES")
    print("="*60)
    print("URL: http://0.0.0.0:8000")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
