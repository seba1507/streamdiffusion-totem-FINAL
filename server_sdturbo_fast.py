#!/usr/bin/env python3
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
torch.backends.cudnn.benchmark = True

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
from PIL import Image
import base64, io, time

app = FastAPI()

print("Loading SD-Turbo...")
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    safety_checker=None
).to("cuda")

# Use TinyVAE for speed
try:
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        torch_dtype=torch.float16
    ).to("cuda")
    print("✅ TinyVAE loaded")
except:
    print("⚠️ Using default VAE")

pipe.set_progress_bar_config(disable=True)
pipe.enable_xformers_memory_efficient_attention()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Warmup
print("Warming up...")
dummy = Image.new('RGB', (512, 512))
for _ in range(3):
    pipe("test", image=dummy, num_inference_steps=1, strength=0.5, guidance_scale=0.0)
print("✅ Ready! Target: 15-25 FPS")

frame_count = 0
last_time = time.time()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SD-Turbo Fast</title>
    <style>
        body { background: #111; color: #0f0; font-family: Arial, sans-serif; margin: 0; padding: 20px; }
        .container { display: flex; gap: 20px; align-items: center; justify-content: center; min-height: 80vh; }
        video, canvas { width: 512px; height: 512px; border: 3px solid #0f0; background: #000; }
        button { background: #0f0; color: #000; border: none; padding: 15px 40px; font-size: 18px; cursor: pointer; margin: 10px; }
        button:hover { background: #0a0; }
        .stats { position: fixed; top: 20px; right: 20px; background: rgba(0,0,0,0.8); padding: 20px; border: 2px solid #0f0; font-size: 18px; }
        .fps { font-size: 32px; font-weight: bold; margin-bottom: 10px; }
        h1 { text-align: center; }
    </style>
</head>
<body>
    <h1>SD-Turbo StreamDiffusion - L40S Optimized</h1>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    <div style="text-align: center;">
        <button onclick="start()">🚀 START</button>
        <button onclick="stop()">⏹ STOP</button>
    </div>
    <div class="stats">
        <div class="fps">FPS: <span id="fps">0</span></div>
        <div>Latency: <span id="latency">0</span>ms</div>
        <div>Model: SD-Turbo (1 step)</div>
    </div>
    
    <script>
        let ws, video, canvas, ctx, streaming = false;
        let frames = 0, lastTime = Date.now();
        
        async function start() {
            video = document.getElementById('video');
            canvas = document.getElementById('output');
            ctx = canvas.getContext('2d');
            canvas.width = canvas.height = 512;
            
            video.srcObject = await navigator.mediaDevices.getUserMedia({video: true});
            
            ws = new WebSocket(`ws://${window.location.host}/ws`);
            ws.onopen = () => { streaming = true; sendFrame(); };
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0, 512, 512);
                img.src = data.image;
                
                document.getElementById('latency').textContent = Math.round(data.time);
                
                frames++;
                const now = Date.now();
                if (now - lastTime > 1000) {
                    document.getElementById('fps').textContent = frames;
                    frames = 0;
                    lastTime = now;
                }
            };
        }
        
        function sendFrame() {
            if (!streaming) return;
            const temp = document.createElement('canvas');
            temp.width = temp.height = 512;
            temp.getContext('2d').drawImage(video, 0, 0, 512, 512);
            ws.send(JSON.stringify({image: temp.toDataURL('image/jpeg', 0.8)}));
            requestAnimationFrame(sendFrame);
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            if (video && video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
        }
    </script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(HTML)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global frame_count, last_time
    
    while True:
        data = await websocket.receive_json()
        img_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        start = time.time()
        
        # SD-Turbo: 1 step = maximum speed
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                result = pipe(
                    prompt="high quality photo",
                    image=image,
                    num_inference_steps=1,
                    strength=0.5,
                    guidance_scale=0.0
                ).images[0]
        
        elapsed = (time.time() - start) * 1000
        
        # FPS logging
        frame_count += 1
        if time.time() - last_time >= 5:
            fps = frame_count / (time.time() - last_time)
            print(f"Average FPS: {fps:.1f}")
            frame_count = 0
            last_time = time.time()
        
        buffered = io.BytesIO()
        result.save(buffered, format="JPEG", quality=85)
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        await websocket.send_json({
            'image': f'data:image/jpeg;base64,{img_str}',
            'time': elapsed
        })

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 SD-TURBO FAST MODE - L40S OPTIMIZED")
    print("Target: 15-25 FPS\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
