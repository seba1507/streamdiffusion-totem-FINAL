#!/usr/bin/env python3
import os
import sys
import time
import threading
import numpy as np
from PIL import Image
import hashlib

# CUDA setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

if not torch.cuda.is_available():
    print("ERROR: CUDA no disponible")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import AutoPipelineForImage2Image, LCMScheduler
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

class DiagnosticDiffusion:
    def __init__(self):
        self.device = "cuda"
        self.dtype = torch.float16
        
        # State
        self.current_task = None
        self.last_result = None
        self.processing = False
        
        # Debug
        self.frame_count = 0
        self.last_input_hash = None
        self.last_output_hash = None
        
        self.init_model()
        
    def init_model(self):
        print("\n🔍 DIAGNOSTIC MODE - Inicializando...")
        
        # Usar LCM que sabemos que funciona
        model_id = "SimianLuo/LCM_Dreamshaper_v7"
        print(f"📥 Cargando {model_id}...")
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)
        
        # LCM Scheduler
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        
        # XFormers
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✅ XFormers habilitado")
        except:
            print("⚠️  XFormers no disponible")
        
        # Simple test
        print("🧪 Test rápido...")
        test_img = Image.new('RGB', (512, 512), (128, 128, 128))
        with torch.no_grad():
            result = self.pipe(
                "test",
                image=test_img,
                num_inference_steps=2,
                strength=0.5,
                guidance_scale=1.0
            ).images[0]
        print("✅ Test completado")
        torch.cuda.empty_cache()
        
        # Start processing
        self.processing = True
        thread = threading.Thread(target=self._process_loop, daemon=True)
        thread.start()
        print("✅ Thread iniciado\n")
    
    def _process_loop(self):
        while self.processing:
            if self.current_task is None:
                time.sleep(0.001)
                continue
            
            try:
                start = time.time()
                task = self.current_task
                self.frame_count += 1
                
                # Get parameters
                image = task['image']
                prompt = task['prompt'] or "portrait"
                strength = float(task['strength'])
                
                # Hash input image
                input_hash = hashlib.md5(np.array(image).tobytes()).hexdigest()[:8]
                
                print(f"\n{'='*60}")
                print(f"🎬 FRAME {self.frame_count}")
                print(f"📝 Prompt: '{prompt}'")
                print(f"💪 Strength: {strength}")
                print(f"🖼️  Input hash: {input_hash}")
                print(f"🔄 Input changed: {input_hash != self.last_input_hash}")
                
                # GENERATE with different parameters to test
                steps = 2 if strength < 0.5 else 3
                cfg = 1.0 if strength < 0.5 else 2.0
                
                print(f"⚙️  Steps: {steps}, CFG: {cfg}")
                
                with torch.no_grad():
                    result = self.pipe(
                        prompt=prompt,
                        image=image,
                        num_inference_steps=steps,
                        strength=strength,
                        guidance_scale=cfg,
                        generator=None  # Random seed
                    ).images[0]
                
                # Hash output
                output_hash = hashlib.md5(np.array(result).tobytes()).hexdigest()[:8]
                
                # Calculate difference
                input_arr = np.array(image)
                output_arr = np.array(result)
                pixel_diff = np.abs(input_arr.astype(float) - output_arr.astype(float)).mean()
                
                elapsed = (time.time() - start) * 1000
                
                print(f"✅ Generated in {elapsed:.0f}ms")
                print(f"📊 Output hash: {output_hash}")
                print(f"📏 Pixel difference: {pixel_diff:.1f}")
                print(f"🔄 Output changed: {output_hash != self.last_output_hash}")
                
                # Warnings
                if pixel_diff < 10:
                    print("⚠️  VERY LOW DIFFERENCE - Output almost identical to input!")
                elif pixel_diff > 100:
                    print("⚠️  VERY HIGH DIFFERENCE - Output very different from input!")
                
                if output_hash == self.last_output_hash:
                    print("❌ OUTPUT IDENTICAL TO LAST FRAME!")
                
                self.last_input_hash = input_hash
                self.last_output_hash = output_hash
                
                # Save result
                self.last_result = {
                    'image': result,
                    'time': elapsed,
                    'debug': {
                        'frame': self.frame_count,
                        'strength': strength,
                        'pixel_diff': pixel_diff,
                        'input_hash': input_hash,
                        'output_hash': output_hash
                    }
                }
                
            except Exception as e:
                print(f"❌ ERROR: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            self.current_task = None
    
    def add_frame(self, image, prompt, strength):
        if self.current_task is not None:
            print("⏭️  Skipping frame (busy)")
            return
        
        self.current_task = {
            'image': image,
            'prompt': prompt,
            'strength': strength
        }
    
    def get_result(self):
        return self.last_result

# Global instance
processor = DiagnosticDiffusion()

# Simple HTML with debug info
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Diagnostic Mode</title>
    <style>
        body {
            background: #111;
            color: #0f0;
            font-family: monospace;
            margin: 0;
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
            border: 2px solid #0f0;
            background: #000;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 20px;
            border: 2px solid #0f0;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }
        
        button {
            padding: 10px 20px;
            background: #0f0;
            color: #000;
            border: none;
            font-weight: bold;
            cursor: pointer;
        }
        
        button:disabled {
            background: #333;
            color: #666;
        }
        
        input[type="range"] {
            width: 100%;
        }
        
        .debug {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0,0,0,0.9);
            padding: 15px;
            border: 2px solid #0f0;
            font-size: 12px;
            min-width: 300px;
        }
        
        .presets {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .preset-btn {
            padding: 5px 10px;
            background: #444;
            color: #fff;
            border: 1px solid #666;
            cursor: pointer;
        }
        
        .preset-btn:hover {
            background: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <canvas id="output"></canvas>
    </div>
    
    <div class="debug">
        <h3>DEBUG INFO</h3>
        <div>FPS: <span id="fps">0</span></div>
        <div>Frame: <span id="frame">0</span></div>
        <div>Strength: <span id="debugStrength">0</span></div>
        <div>Pixel Diff: <span id="pixelDiff">0</span></div>
        <div>Input Hash: <span id="inputHash">-</span></div>
        <div>Output Hash: <span id="outputHash">-</span></div>
        <div>Processing: <span id="processing">0</span>ms</div>
    </div>
    
    <div class="controls">
        <div>
            <button id="startBtn" onclick="start()">START</button>
            <button id="stopBtn" onclick="stop()" disabled>STOP</button>
        </div>
        
        <div>
            <label>Prompt:</label>
            <input type="text" id="prompt" value="cyberpunk style" style="width:100%">
        </div>
        
        <div>
            <label>Strength: <span id="strengthValue">0.5</span></label>
            <input type="range" id="strength" min="0.1" max="0.9" step="0.1" value="0.5" 
                   oninput="document.getElementById('strengthValue').textContent=this.value">
        </div>
        
        <div class="presets">
            <button class="preset-btn" onclick="setPreset(0.1)">0.1</button>
            <button class="preset-btn" onclick="setPreset(0.3)">0.3</button>
            <button class="preset-btn" onclick="setPreset(0.5)">0.5</button>
            <button class="preset-btn" onclick="setPreset(0.7)">0.7</button>
            <button class="preset-btn" onclick="setPreset(0.9)">0.9</button>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let frameCount = 0;
        let lastTime = Date.now();
        
        function setPreset(value) {
            document.getElementById('strength').value = value;
            document.getElementById('strengthValue').textContent = value;
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
                
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('Connected');
                    streaming = true;
                    document.getElementById('startBtn').disabled = true;
                    document.getElementById('stopBtn').disabled = false;
                    sendFrame();
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0);
                        };
                        img.src = data.image;
                        
                        // Update debug info
                        if (data.debug) {
                            document.getElementById('frame').textContent = data.debug.frame;
                            document.getElementById('debugStrength').textContent = data.debug.strength;
                            document.getElementById('pixelDiff').textContent = data.debug.pixel_diff.toFixed(1);
                            document.getElementById('inputHash').textContent = data.debug.input_hash;
                            document.getElementById('outputHash').textContent = data.debug.output_hash;
                            document.getElementById('processing').textContent = Math.round(data.time);
                        }
                        
                        // FPS
                        frameCount++;
                        const now = Date.now();
                        if (now - lastTime >= 1000) {
                            document.getElementById('fps').textContent = frameCount;
                            frameCount = 0;
                            lastTime = now;
                        }
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
                image: tc.toDataURL('image/jpeg', 0.9),
                prompt: document.getElementById('prompt').value,
                strength: parseFloat(document.getElementById('strength').value)
            }));
            
            setTimeout(sendFrame, 200); // 5 FPS for testing
        }
        
        function stop() {
            streaming = false;
            if (ws) ws.close();
            
            const video = document.getElementById('video');
            if (video.srcObject) {
                video.srcObject.getTracks().forEach(t => t.stop());
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
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
    print("🔍 Cliente conectado - DIAGNOSTIC MODE")
    
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
                data.get('strength', 0.5)
            )
            
            result = processor.get_result()
            if result:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=90)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'time': result['time'],
                    'debug': result['debug']
                })
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Cliente desconectado")

if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🔍 DIAGNOSTIC MODE - IMG2IMG TESTING")
    print("="*80)
    print("📊 This will show detailed info about each frame")
    print("🔧 Test different strength values to see the effect")
    print("⚠️  Check console for detailed diagnostics")
    print("🌐 http://0.0.0.0:8000")
    print("="*80)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
