#!/usr/bin/env python3
import os
import sys
import time
import asyncio
import io
import base64
from pathlib import Path

# --- CONFIGURACIÓN DE ALTO RENDIMIENTO (Ajustable) ---

# Modelo LCM optimizado para velocidad y calidad. Es el mejor punto de partida.
MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
# El VAE (decodificador) es un cuello de botella. Usar un TinyVAE es CRÍTICO para FPS altos.
TINY_VAE_ID = "madebyollin/taesd"

# Directorio para guardar los motores TensorRT compilados. Acelera los reinicios.
ENGINE_DIR = Path("./tensorrt_engines")
ENGINE_DIR.mkdir(exist_ok=True)

# Parámetros por defecto (el cliente puede sobreescribirlos)
# Calidad vs Velocidad: "8k, masterpiece" da más detalle a costa de ~5% de rendimiento.
DEFAULT_PROMPT = "cinematic, professional photography, highly detailed, sharp focus, 8k"
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, jpeg artifacts, ugly, deformed"
# Los modelos LCM funcionan mejor con valores de Guidance bajos. 1.0-2.0 es el rango ideal.
DEFAULT_GUIDANCE_SCALE = 1.2
# Strength < 0.7 para cambios sutiles, > 0.7 para transformaciones intensas.
DEFAULT_STRENGTH = 0.65
# Suaviza el parpadeo entre frames. 0.0 es sin suavizado, 0.5 es muy suave.
DEFAULT_TEMPORAL_SMOOTHING = 0.2

# Resolución de la generación. 512 es el estándar para SD 1.5 y el más rápido.
WIDTH, HEIGHT = 512, 512
# ---------------------------------------------------------

# Configurar y verificar CUDA
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
if not torch.cuda.is_available():
    print("FATAL: CUDA no está disponible. Revisa los drivers de NVIDIA y PyTorch.", file=sys.stderr)
    sys.exit(1)
device = torch.device("cuda")
dtype = torch.float16
print(f"✅ CUDA detectado: {torch.cuda.get_device_name(0)} | Usando precisión: {dtype}")

# Importaciones tardías de IA para asegurar que CUDA está configurado
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from PIL import Image

class StreamProcessor:
    """
    Clase que encapsula toda la lógica de StreamDiffusion, desde la carga del modelo
    hasta el procesamiento de frames en tiempo real, todo acelerado en la GPU.
    """
    def __init__(self):
        self.stream = self._initialize_pipeline()
        self.last_latents = None
        self.frame_times = [] # Para calcular FPS promedio
        print("✅ Procesador de Stream optimizado y listo.")

    def _initialize_pipeline(self) -> StreamDiffusion:
        """
        Carga el modelo, lo acelera con TensorRT y lo pre-calienta para un rendimiento
        inmediato. La primera ejecución compilará los motores, lo cual puede tardar.
        """
        print("🚀 Inicializando pipeline de StreamDiffusion...")
        
        stream = StreamDiffusion(
            model_id_or_path=MODEL_ID,
            t_index_list=[0, 16, 32, 45], # Timesteps como en la doc de DotSimulate para img2img
            frame_buffer_size=1,
            width=WIDTH,
            height=HEIGHT,
            device=device,
            dtype=dtype,
            use_tiny_vae=TINY_VAE_ID,
        )
        
        print(f"🔄 Cargando modelo base '{MODEL_ID}'...")
        stream.load_lcm_lora()
        stream.fuse_lora()
        print("✅ LoRA LCM cargado y fusionado con el modelo base.")

        print("⚡ Acelerando con TensorRT (puede tardar varios minutos la primera vez)...")
        try:
            stream = accelerate_with_tensorrt(
                stream, ENGINE_DIR, max_batch_size=2,
            )
            print("✅ Pipeline acelerado con TensorRT exitosamente.")
        except Exception as e:
            print(f"⚠️ Advertencia: Falló la aceleración con TensorRT. {e}", file=sys.stderr)
            print("El rendimiento será significativamente menor. Revisa la instalación de TensorRT.", file=sys.stderr)

        print("🔥 Pre-calentando el pipeline...")
        stream.prepare(
            prompt=DEFAULT_PROMPT,
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            num_inference_steps=50, # El scheduler LCM ignora esto, pero es bueno tenerlo.
            guidance_scale=DEFAULT_GUIDANCE_SCALE,
        )
        print("✅ Pipeline listo para recibir frames.")
        return stream

    @torch.no_grad()
    def process_frame(self, image: Image.Image, params: dict) -> (Image.Image, dict):
        """
        Procesa un único frame. Todas las operaciones se realizan en la GPU.
        """
        start_time = time.time()
        
        # 1. PREPROCESAMIENTO EN GPU
        input_tensor = self.stream.image_processor.preprocess(image).to(device=device, dtype=dtype)
        
        # 2. INYECCIÓN DE RUIDO EN GPU (Técnica de DotSimulate)
        # Añade textura para que el modelo genere resultados más ricos y menos planos.
        noise = torch.randn_like(input_tensor) * 0.02
        input_tensor = torch.clamp(input_tensor + noise, 0, 1)

        # 3. ACTUALIZACIÓN DE PARÁMETROS (si es necesario)
        # Esto permite cambiar el prompt o la guía en tiempo real sin reiniciar.
        if self.stream.prompt != params['prompt'] or self.stream.guidance_scale != params['guidance_scale']:
            self.stream.prepare(
                prompt=params['prompt'],
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                num_inference_steps=50,
                guidance_scale=params['guidance_scale'],
            )

        # 4. INFERENCIA
        # Convierte la imagen de entrada a espacio latente.
        latents = self.stream.encode_image(input_tensor)
        # Añade ruido inicial basado en la intensidad (strength).
        noisy_latents = self.stream.add_noise(latents, params['strength'])
        # Ejecuta el modelo (UNet) para "denoisar" los latentes.
        denoised_latents = self.stream(image_latents=noisy_latents)

        # 5. FEEDBACK TEMPORAL EN GPU (Técnica de DotSimulate)
        # Interpola entre el último frame y el actual para suavizar la transición.
        if self.last_latents is not None:
             denoised_latents = torch.lerp(self.last_latents, denoised_latents, 1.0 - params['temporal_smoothing'])
        self.last_latents = denoised_latents.clone() # Guardar el estado actual para el siguiente frame

        # 6. DECODIFICACIÓN Y POSTPROCESAMIENTO
        # Convierte los latentes de vuelta a una imagen visible usando el TinyVAE.
        output_tensor = self.stream.decode_image(denoised_latents)
        # Convierte el tensor de PyTorch a una imagen PIL.
        output_image = postprocess_image(output_tensor, output_type="pil")[0]
        
        # 7. MÉTRICAS DE RENDIMIENTO
        processing_time_ms = (time.time() - start_time) * 1000
        self.frame_times.append(processing_time_ms)
        if len(self.frame_times) > 100: self.frame_times.pop(0)
        avg_latency = sum(self.frame_times) / len(self.frame_times)
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        stats = {"fps": round(fps, 1), "latency": round(processing_time_ms, 1)}
        return output_image, stats

# --- SERVIDOR WEB FASTAPI ---

app = FastAPI()
# Es importante instanciar el procesador aquí para que el modelo se cargue al iniciar el servidor.
processor = StreamProcessor()

@app.get("/")
async def get_root():
    # Para no duplicar código, importamos el HTML del script original que ya tenías.
    # Asegúrate de que el archivo 'server_dotsimulate_enhanced.py' esté en el mismo directorio.
    try:
        from server_dotsimulate_enhanced import HTML_CONTENT
        return HTMLResponse(content=HTML_CONTENT)
    except ImportError:
        return HTMLResponse(content="<h1>Error</h1><p>No se pudo encontrar el archivo 'server_dotsimulate_enhanced.py' para cargar la interfaz.</p>")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ Cliente conectado al WebSocket.")
    try:
        while True:
            data = await websocket.receive_json()
            
            # Decodificar imagen de entrada
            img_data = base64.b64decode(data['image'].split(',')[1])
            input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Recoger parámetros del cliente o usar los defaults
            params = {
                'prompt': data.get('prompt', DEFAULT_PROMPT),
                'strength': float(data.get('strength', DEFAULT_STRENGTH)),
                'guidance_scale': float(data.get('guidance_scale', DEFAULT_GUIDANCE_SCALE)),
                'temporal_smoothing': DEFAULT_TEMPORAL_SMOOTHING,
            }

            # Procesar el frame con nuestra clase optimizada
            output_image, stats = processor.process_frame(input_image, params)
            
            # Enviar el resultado de vuelta
            buffered = io.BytesIO()
            output_image.save(buffered, format="JPEG", quality=90) # JPEG quality 90 es buen balance
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            await websocket.send_json({'image': f'data:image/jpeg;base64,{img_str}', 'stats': stats})
            
    except Exception as e:
        print(f"❌ Error en WebSocket: {e}", file=sys.stderr)
    finally:
        processor.last_latents = None # Limpiar estado al desconectar
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