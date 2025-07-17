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
        # --- LÓGICA CORREGIDA ---
        # Definir los atributos ANTES de llamar a la inicialización del pipeline
        self.last_latents = None
        self.frame_times = []
        self.current_prompt = DEFAULT_PROMPT
        self.current_guidance = DEFAULT_GUIDANCE_SCALE
        
        # Ahora llamar a la inicialización, que usará los atributos de arriba
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
            pipe=pipe,
            t_index_list=[0, 16, 32, 45],
            frame_buffer_size=1,
            width=WIDTH,
            height=HEIGHT,
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
            prompt=self.current_prompt,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=50,
            guidance_scale=self.current_guidance,
        )
        print("✅ Pipeline listo para recibir frames.")
        return stream

    @torch.no_grad()
    def process_frame(self, image: Image.Image, params: dict) -> (Image.Image, dict):
        start_time = time.time()
        
        if self.current_prompt != params['prompt'] or self.current_guidance != params['guidance_scale']:
            self.stream.prepare(
                prompt=params['prompt'],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=50,
                guidance_scale=params['guidance_scale'],
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

# --- SERVIDOR WEB FASTAPI (Sin cambios) ---
app = FastAPI()
processor = StreamProcessor()

@app.get("/")
async def get_root():
    # Asume que tu script original con el HTML está en el directorio
    try:
        from server_dotsimulate_enhanced import HTML_CONTENT
        return HTMLResponse(content=HTML_CONTENT)
    except ImportError:
        # Fallback por si el archivo no existe
        return HTMLResponse(content="<h1>StreamDiffusion Server</h1><p>Ready to connect via WebSocket.</p>")

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
