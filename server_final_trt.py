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
    Wrapper minimalista para ejecutar planes .engine con 1..N inputs / 1 output.
    Supone que los bindings 0..N‑1 son inputs y el último es output.
    """
    def __init__(self, path: str):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine  = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.bindings = [None] * self.engine.num_bindings
        self.out = None
        self.stream = torch.cuda.current_stream().cuda_stream
        print(f"📦 TRT cargado: {Path(path).name}")

    def __call__(self, *inputs: torch.Tensor) -> torch.Tensor:
        assert len(inputs) == self.engine.num_bindings - 1, "Nº de inputs != bindings‑1"
        for idx, t in enumerate(inputs):
            self.bindings[idx] = int(t.data_ptr())
        if self.out is None:
            out_shape = tuple(self.engine.get_binding_shape(self.engine.num_bindings - 1))
            self.out = torch.empty(out_shape, dtype=inputs[0].dtype, device=inputs[0].device)
            self.bindings[-1] = int(self.out.data_ptr())
        self.context.execute_async_v2(self.bindings, self.stream)
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
            t_index_list=[0, 16, 32, 45],
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
            return self.trt_unet(latent_in, t, enc_hidden)
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
async def root(): return HTMLResponse(content="<h1>StreamDiffusion TOTEM (usar frontend)…</h1>")

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
