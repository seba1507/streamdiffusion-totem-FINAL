#!/usr/bin/env python3
"""
Enhanced StreamDiffusion Server - Optimized for 15-20 FPS
Based on research findings for NVIDIA L40S optimization
"""
import os
import sys
import time
import asyncio
import threading
import queue
from collections import deque
import hashlib
import numpy as np
from PIL import Image
import cv2
import gc
import warnings
warnings.filterwarnings("ignore")

# L40S-specific CUDA optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

import torch
import torch.nn.functional as F

# Verify CUDA
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

print(f"✅ CUDA OK: {torch.cuda.get_device_name(0)}")
print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from diffusers import (
    AutoPipelineForImage2Image, 
    LCMScheduler, 
    AutoencoderTiny,
    StableDiffusionPipeline
)
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
import base64
import io
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizedStreamDiffusion:
    def __init__(self):
        self.pipe = None
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Processing queues with smaller size for lower latency
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Optimized similarity filter
        self.last_image_hash = None
        self.similarity_threshold = 0.98  # Higher threshold as per research
        self.frame_skip_counter = 0
        self.max_skip_frames = 10
        
        # Performance tracking
        self.total_frames = 0
        self.skipped_frames = 0
        self.processing_times = deque(maxlen=30)
        
        # CUDA stream for async operations
        self.cuda_stream = torch.cuda.Stream()
        
        # Memory pool for frame data
        self.frame_buffer_pool = []
        
        self.init_model()
        
    def init_model(self):
        """Initialize optimized model pipeline"""
        logger.info("🚀 Initializing Optimized StreamDiffusion Pipeline...")
        
        # Model selection: SD-Turbo for maximum speed with good quality
        model_id = "stabilityai/sd-turbo"
        
        # Load pipeline with optimizations
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True,
            variant="fp16"
        ).to(self.device)
        
        # CRITICAL: Use LCM Scheduler for 4-step generation
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config,
            beta_schedule="scaled_linear",
            timestep_spacing="trailing"
        )
        
        # Load LCM-LoRA for acceleration
        try:
            logger.info("Loading LCM-LoRA...")
            self.pipe.load_lora_weights(
                "latent-consistency/lcm-lora-sdv1-5",
                adapter_name="lcm"
            )
            self.pipe.set_adapters(["lcm"], adapter_weights=[1.0])
            logger.info("✅ LCM-LoRA loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ LCM-LoRA loading failed: {e}")
        
        # Use Tiny AutoEncoder for faster VAE operations
        try:
            logger.info("Loading Tiny AutoEncoder...")
            self.pipe.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype
            ).to(self.device)
            logger.info("✅ TAESD loaded - 3-4x faster VAE")
        except Exception as e:
            logger.warning(f"⚠️ TAESD loading failed, using default VAE: {e}")
        
        # Pipeline optimizations
        self.pipe.set_progress_bar_config(disable=True)
        
        # Enable memory efficient attention (XFormers alternative)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            logger.info("✅ XFormers memory efficient attention enabled")
        except:
            self.pipe.enable_attention_slicing(1)
            logger.info("✅ Attention slicing enabled")
        
        # Enable VAE slicing and tiling for memory efficiency
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        
        # torch.compile optimization for UNet (if available)
        if hasattr(torch, 'compile'):
            try:
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                logger.info("✅ UNet compiled with torch.compile")
            except:
                logger.info("⚠️ torch.compile not available")
        
        # Pre-warm the pipeline
        self._warmup_pipeline()
        
        # Start processing thread
        self.start_processing_thread()
    
    def _warmup_pipeline(self):
        """Pre-warm pipeline to avoid initial latency"""
        logger.info("Pre-warming pipeline...")
        dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i in range(2):
                    _ = self.pipe(
                        prompt="warmup",
                        image=dummy_image,
                        num_inference_steps=4,
                        strength=0.5,
                        guidance_scale=1.0
                    ).images[0]
        
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("✅ Pipeline warmed up")
    
    def calculate_image_similarity_fast(self, image):
        """Ultra-fast similarity calculation"""
        try:
            # Downsample more aggressively for speed
            small_img = image.resize((16, 16), Image.Resampling.NEAREST)
            img_array = np.array(small_img, dtype=np.uint8)
            return hashlib.md5(img_array.tobytes()).hexdigest()
        except:
            return str(time.time())
    
    def should_skip_frame(self, image):
        """Optimized frame skipping logic"""
        if self.frame_skip_counter >= self.max_skip_frames:
            self.frame_skip_counter = 0
            return False
            
        current_hash = self.calculate_image_similarity_fast(image)
        
        if self.last_image_hash and current_hash == self.last_image_hash:
            self.frame_skip_counter += 1
            self.skipped_frames += 1
            return True
        
        self.last_image_hash = current_hash
        self.frame_skip_counter = 0
        return False
    
    def preprocess_image_fast(self, image):
        """Minimal preprocessing for speed"""
        # Only resize if needed
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        return image
    
    def start_processing_thread(self):
        """Start optimized processing thread"""
        self.processing = True
        thread = threading.Thread(target=self._optimized_processing_loop, daemon=True)
        thread.start()
        logger.info("✅ Processing thread started")
    
    def _optimized_processing_loop(self):
        """Main processing loop with all optimizations"""
        while self.processing:
            try:
                # Get frame with short timeout
                frame_data = self.frame_queue.get(timeout=0.05)
                
                # Skip similar frames
                if self.should_skip_frame(frame_data['image']):
                    # Reuse last result for similar frames
                    try:
                        if self.result_queue.qsize() > 0:
                            last_result = list(self.result_queue.queue)[-1]
                            last_result['skipped'] = True
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                start_time = time.time()
                
                # Preprocess image
                processed_image = self.preprocess_image_fast(frame_data['image'])
                
                # Optimized generation parameters
                prompt = frame_data.get('prompt', 'high quality photo')
                strength = frame_data.get('strength', 0.5)  # Lower default for speed
                
                # CRITICAL: LCM-optimized parameters
                guidance_scale = 1.5  # LCM optimal range: 1.0-2.0
                num_steps = 4  # Sweet spot for quality/speed
                
                # Generate with optimizations
                with torch.cuda.stream(self.cuda_stream):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            result = self.pipe(
                                prompt=prompt,
                                image=processed_image,
                                num_inference_steps=num_steps,
                                strength=strength,
                                guidance_scale=guidance_scale,
                                output_type="pil",
                                return_dict=True
                            ).images[0]
                
                # Sync stream
                self.cuda_stream.synchronize()
                
                processing_time = (time.time() - start_time) * 1000
                self.processing_times.append(processing_time)
                self.total_frames += 1
                
                # Prepare result
                result_data = {
                    'image': result,
                    'processing_time': processing_time,
                    'timestamp': frame_data['timestamp'],
                    'stats': {
                        'total_frames': self.total_frames,
                        'skipped_frames': self.skipped_frames,
                        'skip_rate': (self.skipped_frames / max(1, self.total_frames)) * 100,
                        'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
                        'strength': strength,
                        'steps': num_steps,
                        'guidance_scale': guidance_scale
                    }
                }
                
                # Non-blocking queue put
                try:
                    # Remove oldest if queue is full
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result_data)
                except:
                    pass
                
                if self.total_frames % 30 == 0:
                    logger.info(f"Performance: {1000 / np.mean(self.processing_times):.1f} FPS avg")
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing error: {e}")
                continue
    
    def add_frame(self, image, prompt, strength, guidance_scale, timestamp):
        """Add frame with non-blocking logic"""
        frame_data = {
            'image': image,
            'prompt': prompt,
            'strength': strength,
            'guidance_scale': guidance_scale,
            'timestamp': timestamp
        }
        
        try:
            # Non-blocking put with queue management
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()  # Drop oldest frame
                except:
                    pass
            self.frame_queue.put_nowait(frame_data)
        except:
            pass
    
    def get_latest_result(self):
        """Get most recent result"""
        result = None
        
        # Get all available results and return the latest
        while not self.result_queue.empty():
            try:
                result = self.result_queue.get_nowait()
            except:
                break
                
        return result
    
    def cleanup(self):
        """Cleanup resources"""
        self.processing = False
        torch.cuda.empty_cache()
        gc.collect()

# Global processor instance
processor = None

# Optimized HTML interface
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>StreamDiffusion Optimized - Brand Activation</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            background: #0a0a0a;
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
        }
        
        .container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 40px;
            padding: 20px;
        }
        
        video, canvas {
            width: 512px;
            height: 512px;
            border: 2px solid #00ff88;
            background: #000;
            border-radius: 16px;
            box-shadow: 0 0 40px rgba(0, 255, 136, 0.3);
        }
        
        .output-container {
            position: relative;
        }
        
        .controls {
            position: fixed;
            bottom: 20px;
            left: 20px;
            right: 20px;
            background: rgba(20, 20, 20, 0.95);
            padding: 30px;
            border-radius: 24px;
            border: 1px solid #333;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 30px;
            backdrop-filter: blur(20px);
            z-index: 1000;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        
        .control-label {
            color: #00ff88;
            font-weight: 600;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        
        button {
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            background: #00ff88;
            color: #000;
            border: none;
            border-radius: 12px;
            transition: all 0.2s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        button:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 255, 136, 0.4);
        }
        
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        .preset-btn {
            background: #444;
            color: #fff;
            padding: 12px 20px;
            font-size: 14px;
        }
        
        .preset-btn.active {
            background: #00ff88;
            color: #000;
        }
        
        input[type="range"] {
            width: 100%;
            height: 6px;
            -webkit-appearance: none;
            background: #333;
            outline: none;
            border-radius: 3px;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00ff88;
            border-radius: 50%;
            cursor: pointer;
        }
        
        textarea {
            padding: 14px;
            font-size: 14px;
            border-radius: 12px;
            background: #1a1a1a;
            color: #fff;
            border: 1px solid #333;
            font-family: inherit;
            resize: none;
            height: 80px;
        }
        
        textarea:focus {
            outline: none;
            border-color: #00ff88;
        }
        
        .range-value {
            color: #00ff88;
            font-weight: 600;
            text-align: center;
            font-size: 16px;
            margin-top: 6px;
        }
        
        .stats {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(20, 20, 20, 0.95);
            padding: 24px;
            border-radius: 16px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 13px;
            border: 1px solid #333;
            min-width: 280px;
            backdrop-filter: blur(20px);
        }
        
        .stat-value {
            color: #00ff88;
            font-weight: 600;
        }
        
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding-bottom: 10px;
            border-bottom: 1px solid #222;
        }
        
        .status-indicator {
            position: absolute;
            top: 10px;
            left: 10px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #ff4444;
            transition: all 0.3s;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.active {
            background: #00ff88;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .brand-header {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 24px;
            font-weight: 700;
            color: #00ff88;
            text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
        }
        
        .preset-container {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .performance-indicator {
            width: 100%;
            height: 4px;
            background: #333;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .performance-bar {
            height: 100%;
            background: #00ff88;
            width: 0%;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="brand-header">StreamDiffusion Pro</div>
    
    <div class="container">
        <video id="video" autoplay muted playsinline></video>
        <div class="output-container">
            <canvas id="output"></canvas>
            <div class="status-indicator" id="statusIndicator"></div>
        </div>
    </div>
    
    <div class="stats">
        <div style="text-align: center; font-weight: 700; color: #00ff88; margin-bottom: 16px;">
            PERFORMANCE METRICS
        </div>
        <div class="stat-row">
            <span>FPS:</span>
            <span id="fps" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Latency:</span>
            <span id="latency" class="stat-value">0</span>ms
        </div>
        <div class="stat-row">
            <span>Frames:</span>
            <span id="totalFrames" class="stat-value">0</span>
        </div>
        <div class="stat-row">
            <span>Skip Rate:</span>
            <span id="skipRate" class="stat-value">0</span>%
        </div>
        <div class="stat-row">
            <span>Target FPS:</span>
            <span class="stat-value">15-20</span>
        </div>
        <div class="performance-indicator">
            <div class="performance-bar" id="performanceBar"></div>
        </div>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <div class="control-label">System Control</div>
            <button id="startBtn" onclick="start()">🚀 START EXPERIENCE</button>
            <button id="stopBtn" onclick="stop()" disabled>⏹ STOP</button>
        </div>
        
        <div class="control-group">
            <div class="control-label">Style Presets</div>
            <div class="preset-container">
                <button class="preset-btn active" onclick="setPreset('brand')">Brand Look</button>
                <button class="preset-btn" onclick="setPreset('dynamic')">Dynamic</button>
                <button class="preset-btn" onclick="setPreset('artistic')">Artistic</button>
                <button class="preset-btn" onclick="setPreset('minimal')">Minimal</button>
            </div>
        </div>
        
        <div class="control-group">
            <div class="control-label">Custom Prompt</div>
            <textarea id="customPrompt" placeholder="Enter style description...">high quality professional photo, brand activation style</textarea>
        </div>
        
        <div class="control-group">
            <div class="control-label">Transformation Strength</div>
            <input type="range" id="strengthSlider" min="0.3" max="0.7" step="0.05" value="0.5" oninput="updateStrengthValue()">
            <div class="range-value" id="strengthValue">0.50</div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let streaming = false;
        let video = null;
        let canvas = null;
        let ctx = null;
        
        let frameCount = 0;
        let lastTime = Date.now();
        let latencies = [];
        let targetFPS = 20;
        
        // WebSocket configuration optimized for low latency
        const WS_CONFIG = {
            reconnectInterval: 1000,
            maxReconnectAttempts: 5,
            pingInterval: 30000,
            messageTimeout: 5000
        };
        
        // Style presets optimized for speed
        const PRESETS = {
            brand: {
                prompt: "high quality professional photo, brand activation style",
                strength: 0.5
            },
            dynamic: {
                prompt: "dynamic energetic style, vibrant colors, motion",
                strength: 0.6
            },
            artistic: {
                prompt: "artistic interpretation, creative style",
                strength: 0.65
            },
            minimal: {
                prompt: "minimal clean style, simple aesthetic",
                strength: 0.4
            }
        };
        
        let currentPreset = 'brand';
        
        function setPreset(preset) {
            currentPreset = preset;
            const config = PRESETS[preset];
            document.getElementById('customPrompt').value = config.prompt;
            document.getElementById('strengthSlider').value = config.strength;
            updateStrengthValue();
            
            // Update UI
            document.querySelectorAll('.preset-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
        }
        
        function updateStrengthValue() {
            const value = parseFloat(document.getElementById('strengthSlider').value);
            document.getElementById('strengthValue').textContent = value.toFixed(2);
        }
        
        async function start() {
            try {
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d', { alpha: false });
                canvas.width = 512;
                canvas.height = 512;
                
                // Request camera with optimal settings
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 },
                        facingMode: 'user'
                    }
                });
                video.srcObject = stream;
                
                await new Promise(resolve => {
                    video.onloadedmetadata = resolve;
                });
                
                connectWebSocket();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Camera access error: ' + error.message);
            }
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            // Configure WebSocket for low latency
            ws.binaryType = 'arraybuffer';
            
            ws.onopen = () => {
                console.log('Connected to StreamDiffusion server');
                streaming = true;
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('statusIndicator').classList.add('active');
                
                // Start frame sending
                sendFrame();
                
                // Start ping to prevent timeout
                startPing();
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.image) {
                        const img = new Image();
                        img.onload = () => {
                            ctx.drawImage(img, 0, 0, 512, 512);
                        };
                        img.src = data.image;
                        updateStats(data);
                    }
                } catch (e) {
                    console.error('Message parse error:', e);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from server');
                if (streaming) {
                    // Attempt reconnection
                    setTimeout(connectWebSocket, WS_CONFIG.reconnectInterval);
                }
            };
        }
        
        let pingInterval = null;
        function startPing() {
            pingInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, WS_CONFIG.pingInterval);
        }
        
        function sendFrame() {
            if (!streaming || !ws || ws.readyState !== WebSocket.OPEN) return;
            
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                requestAnimationFrame(sendFrame);
                return;
            }
            
            // Create temporary canvas for capturing frame
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d', { alpha: false });
            
            // Center crop video to square
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            // Get current settings
            const prompt = document.getElementById('customPrompt').value || PRESETS[currentPreset].prompt;
            const strength = parseFloat(document.getElementById('strengthSlider').value);
            
            // Convert to base64 with optimal quality
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.85);
            
            // Send frame data
            try {
                ws.send(JSON.stringify({
                    image: imageData,
                    prompt: prompt,
                    strength: strength,
                    guidance_scale: 1.5,  // Optimized for LCM
                    timestamp: Date.now()
                }));
            } catch (e) {
                console.error('Send error:', e);
            }
            
            // Schedule next frame
            requestAnimationFrame(sendFrame);
        }
        
        function updateStats(data) {
            frameCount++;
            
            if (data.processing_time) {
                latencies.push(data.processing_time);
                if (latencies.length > 30) latencies.shift();
            }
            
            const now = Date.now();
            if (now - lastTime >= 1000) {
                const fps = frameCount;
                document.getElementById('fps').textContent = fps;
                
                // Update performance bar
                const performanceRatio = Math.min(fps / targetFPS, 1.0);
                document.getElementById('performanceBar').style.width = (performanceRatio * 100) + '%';
                
                frameCount = 0;
                lastTime = now;
                
                if (latencies.length > 0) {
                    const avgLatency = latencies.reduce((a, b) => a + b) / latencies.length;
                    document.getElementById('latency').textContent = Math.round(avgLatency);
                }
            }
            
            if (data.stats) {
                document.getElementById('totalFrames').textContent = data.stats.total_frames;
                document.getElementById('skipRate').textContent = data.stats.skip_rate.toFixed(1);
            }
        }
        
        function stop() {
            streaming = false;
            
            if (pingInterval) {
                clearInterval(pingInterval);
                pingInterval = null;
            }
            
            if (ws) {
                ws.close();
                ws = null;
            }
            
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
                video.srcObject = null;
            }
            
            if (ctx) {
                ctx.clearRect(0, 0, 512, 512);
            }
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('statusIndicator').classList.remove('active');
        }
        
        // Cleanup on page unload
        window.addEventListener('beforeunload', stop);
        
        // Initialize UI
        updateStrengthValue();
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
    client_address = websocket.client.host
    logger.info(f"Client connected: {client_address}")
    
    try:
        while True:
            # Receive with timeout to handle disconnections
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=60.0  # 60 second timeout
                )
            except asyncio.TimeoutError:
                logger.warning("WebSocket receive timeout")
                break
            
            # Handle ping messages
            if data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                continue
            
            # Validate frame data
            if 'image' not in data:
                continue
            
            try:
                # Decode image
                img_data = base64.b64decode(data['image'].split(',')[1])
                input_image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
            except Exception as e:
                logger.error(f"Image decode error: {e}")
                continue
            
            # Get parameters
            prompt = data.get('prompt', 'high quality photo')
            strength = data.get('strength', 0.5)
            guidance_scale = data.get('guidance_scale', 1.5)
            
            # Add frame to processor
            processor.add_frame(
                input_image, 
                prompt,
                strength,
                guidance_scale,
                data.get('timestamp', time.time() * 1000)
            )
            
            # Get latest result
            result = processor.get_latest_result()
            
            if result:
                # Convert to base64
                buffered = io.BytesIO()
                result['image'].save(buffered, format="JPEG", quality=85, optimize=True)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # Send result
                await websocket.send_json({
                    'image': f'data:image/jpeg;base64,{img_str}',
                    'processing_time': result['processing_time'],
                    'timestamp': result['timestamp'],
                    'stats': result['stats']
                })
            
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {client_address}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info(f"Closing connection: {client_address}")

@app.on_event("startup")
async def startup_event():
    global processor
    processor = OptimizedStreamDiffusion()
    logger.info("StreamDiffusion server started")

@app.on_event("shutdown")
async def shutdown_event():
    global processor
    if processor:
        processor.cleanup()
    logger.info("StreamDiffusion server stopped")

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*80)
    print("🚀 STREAMDIFFUSION OPTIMIZED FOR BRAND ACTIVATION")
    print("="*80)
    print("✅ Target Performance: 15-20 FPS")
    print("✅ Model: SD-Turbo with LCM-LoRA")
    print("✅ Optimizations: TinyVAE, XFormers, CUDA Graphs")
    print("✅ Resolution: 512x512")
    print("🌐 URL: http://0.0.0.0:8000")
    print("="*80 + "\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        log_level="info",
        access_log=False  # Disable access logs for performance
    )
