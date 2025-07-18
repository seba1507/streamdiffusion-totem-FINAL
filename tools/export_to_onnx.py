#!/usr/bin/env python3
import torch, os
from diffusers import AutoPipelineForImage2Image

MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
OUT = "onnx"                     # carpeta de salida

os.makedirs(OUT, exist_ok=True)
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16
).to("cuda")


# ------------- UNet -------------
unet = pipe.unet.half().to("cuda")

dummy_latent   = torch.randn(1, 4, 64, 64, dtype=torch.float16, device="cuda")
dummy_timestep = torch.tensor([0], dtype=torch.float32, device="cuda")

# Embeddings de texto dummy: (batch=1, tokens=77, dim=768)
dummy_embed = torch.randn(1, 77, 768, dtype=torch.float16, device="cuda")

torch.onnx.export(
    unet,
    (dummy_latent, dummy_timestep, dummy_embed),   # ← 3 entradas
    f"{OUT}/unet.onnx",
    input_names=["latent", "timestep", "enc_hidden"],
    output_names=["noise_pred"],
    opset_version=17,
    dynamic_axes={
        "latent":     {0: "b", 2: "h", 3: "w"},
        "enc_hidden": {0: "b", 1: "s"},
        "noise_pred": {0: "b", 2: "h", 3: "w"},
    },
)

# ------------- VAE Encoder -------------
encoder = pipe.vae.encoder.half().to("cuda")
dummy_rgb = torch.randn(1, 3, 512, 512, dtype=torch.float16, device="cuda")
torch.onnx.export(
    encoder,
    dummy_rgb,
    f"{OUT}/vae_encoder.onnx",
    input_names=["img"],
    output_names=["latent"],
    opset_version=17,
    dynamic_axes={
        "img": {0: "b", 2: "h", 3: "w"},
        "latent": {0: "b", 2: "h", 3: "w"},
    },
)

# ------------- VAE Decoder -------------
decoder = pipe.vae.decoder.half().to("cuda")
torch.onnx.export(
    decoder,
    dummy_latent,
    f"{OUT}/vae_decoder.onnx",
    input_names=["latent"],
    output_names=["img"],
    opset_version=17,
    dynamic_axes={
        "latent": {0: "b", 2: "h", 3: "w"},
        "img": {0: "b", 2: "h", 3: "w"},
    },
)

print("✅ ONNX exportado en", OUT)
