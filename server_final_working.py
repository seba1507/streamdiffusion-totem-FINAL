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

# HTML simplificado
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>StreamDiffusion Working</title>
    <style>
        body { background: #1a1a1a; color: white; font-family: monospace; padding: 20px; }
        video, canvas { width: 512px; height: 512px; border: 2px solid #00ff88; margin: 10px; }
        button { padding: 10px 20px; margin: 10px; font-size: 16px; }
        #status { color: #00ff88; margin: 10px; font-size: 20px; }
        .container { display: flex; gap: 20px; }
    </style>
</head>
<body>
    <h1>StreamDiffusion TensorRT - L40S</h1>
    <div id="status">Ready to start</div>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    <button onclick="start()">Start Stream</button>
    <button onclick="stop()">Stop</button>
    <script>
        let ws = null, streaming = false;
        
        async function start() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('output');
            const ctx = canvas.getContext('2d');
            canvas.width = 512; canvas.height = 512;
            
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 1280, height: 720 } 
            });
            video.srcObject = stream;
            
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => {
                document.getElementById('status').textContent = 'Connected - Streaming';
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
            
            // Crop center square
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            ws.send(JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.9)
            }));
            
            requestAnimationFrame(sendFrame);
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            const video = document.getElementById('video');
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
            }
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

class StreamProcessor:
    def __init__(self):
        print("🚀 Inicializando StreamDiffusion...")
        
        # Cargar pipeline SIN variant="fp16"
        pipe = AutoPipelineForImage2Image.from_pretrained(
            MODEL_ID, 
            torch_dtype=dtype,
            use_safetensors=True,
            safety_checker=None
        )
        
        # TinyVAE
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE_ID, torch_dtype=dtype)
        pipe = pipe.to(device)
        
        # StreamDiffusion simple
        self.stream = StreamDiffusion(
            pipe=pipe,
            t_index_list=[32, 45],
            torch_dtype=dtype,
            width=WIDTH,
            height=HEIGHT,
            do_add_noise=True,
            frame_buffer_size=1,
            use_denoising_batch=True,
            cfg_type="none",
        )
        
        # LCM-LoRA
        self.stream.load_lcm_lora()
        self.stream.fuse_lora()
        
        # TensorRT
        try:
            self.stream = accelerate_with_tensorrt(
                self.stream, 
                ENGINE_DIR, 
                max_batch_size=1
            )
            print("✅ TensorRT acelerado")
        except Exception as e:
            print(f"⚠️ TensorRT: {e}")
        
        # Preparar
        self.stream.prepare(
            prompt="beautiful scenery, high quality",
            num_inference_steps=50,
            guidance_scale=1.0,
        )
        
        print("✅ Listo para streaming")
        self.frame_count = 0
        self.start_time = time.time()
    
    @torch.no_grad()
    def process(self, pil_image):
        # Resize si es necesario
        if pil_image.size != (WIDTH, HEIGHT):
            pil_image = pil_image.resize((WIDTH, HEIGHT), Image.LANCZOS)
        
        # Procesar
        output = self.stream(pil_image)
        
        # Postprocesar
        if isinstance(output, torch.Tensor):
            output_image = postprocess_image(output, output_type="pil")[0]
        else:
            output_image = output
        
        # FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        return output_image, round(fps, 1)

# FastAPI
app = FastAPI()
processor = StreamProcessor()

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
            
            # Decodificar
            img_data = base64.b64decode(data['image'].split(',')[1])
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Procesar
            output_image, fps = processor.process(pil_image)
            
            # Codificar
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
    print("STREAMDIFFUSION CON TENSORRT")
    print("="*60)
    print("🌐 http://0.0.0.0:8000")
    print("🏃 RunPod: https://9tpw6lrh5yp5ll-8000.proxy.runpod.net/")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
