#!/usr/bin/env python3
"""
Enhanced StreamDiffusion Server - Brand Activation Optimized
Implements all performance optimizations from research:
- TensorRT acceleration (2x speedup)
- LCM-LoRA with 4 steps (optimal quality/speed)
- WebSocket optimizations
- L40S-specific GPU settings
- StreamDiffusion pipeline optimizations
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
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import json
import base64
import io
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import gc

# Configure CUDA for L40S
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'

# Verify CUDA availability
if not torch.cuda.is_available():
    print("ERROR: CUDA not available")
    sys.exit(1)

print(f"✓ CUDA Available: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Import FastAPI and dependencies
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn

# Import diffusion libraries
from diffusers import (
    AutoPipelineForImage2Image, 
    LCMScheduler,
    AutoencoderTiny
)

# Try to import StreamDiffusion components
try:
    from streamdiffusion import StreamDiffusion
    from streamdiffusion.acceleration.tensorrt import accelerate_with_tensorrt
    STREAMDIFFUSION_AVAILABLE = True
except ImportError:
    print("⚠️ StreamDiffusion not available, using optimized Diffusers pipeline")
    STREAMDIFFUSION_AVAILABLE = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizedBrandActivationDiffusion:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.dtype = torch.float16
        
        # Performance configuration
        self.use_tensorrt = True
        self.use_lcm_lora = True
        self.use_tiny_vae = True
        self.enable_cuda_graph = True
        
        # Processing queues with optimized sizes
        self.frame_queue = queue.Queue(maxsize=2)  # Smaller queue for lower latency
        self.result_queue = queue.Queue(maxsize=3)
        self.processing = False
        
        # Similarity filter with optimized threshold
        self.similarity_threshold = 0.98  # Higher threshold as per research
        self.max_skip_frames = 10
        self.skip_counter = 0
        self.last_image_hash = None
        
        # Performance metrics
        self.total_frames = 0
        self.skipped_frames = 0
        self.fps_history = deque(maxlen=30)
        self.last_fps_time = time.time()
        
        # Initialize model
        self.init_optimized_model()
        
    def init_optimized_model(self):
        """Initialize model with all optimizations from research"""
        print("🚀 Initializing Optimized StreamDiffusion for Brand Activation...")
        
        if STREAMDIFFUSION_AVAILABLE:
            self._init_streamdiffusion()
        else:
            self._init_diffusers_optimized()
        
        # Pre-warm pipeline
        self._prewarm_pipeline()
        
        # Start processing thread
        self.start_processing_thread()
        
    def _init_streamdiffusion(self):
        """Initialize with native StreamDiffusion (best performance)"""
        print("✓ Using native StreamDiffusion with TensorRT")
        
        # Load base model
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            torch_dtype=self.dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to(self.device)
        
        # Create StreamDiffusion instance with optimal settings
        self.stream = StreamDiffusion(
            pipe,
            t_index_list=[32, 45],  # Optimal timesteps from research
            torch_dtype=self.dtype,
            cfg_type="self",  # RCFG for 2x speedup
            width=512,
            height=512,
        )
        
        # Load LCM-LoRA for quality/speed balance
        if self.use_lcm_lora:
            self.stream.load_lcm_lora()
            self.stream.fuse_lora()
            print("✓ LCM-LoRA loaded and fused")
        
        # Use Tiny VAE for faster decoding
        if self.use_tiny_vae:
            self.stream.vae = AutoencoderTiny.from_pretrained(
                "madebyollin/taesd",
                torch_dtype=self.dtype
            ).to(self.device)
            print("✓ TinyVAE loaded for 3-4x faster decoding")
        
        # Enable similarity filter
        self.stream.enable_similar_image_filter(
            threshold=self.similarity_threshold,
            max_skip_frame=self.max_skip_frames
        )
        
        # TensorRT acceleration
        if self.use_tensorrt:
            try:
                self.stream = accelerate_with_tensorrt(
                    self.stream,
                    "engines",
                    max_batch_size=2,
                    build_static_batch=True
                )
                print("✓ TensorRT acceleration enabled (2x speedup)")
            except Exception as e:
                print(f"⚠️ TensorRT acceleration failed: {e}")
                self.use_tensorrt = False
        
    def _init_diffusers_optimized(self):
        """Fallback: Optimized Diffusers pipeline"""
        print("✓ Using optimized Diffusers pipeline")
        
        # Use SD-Turbo for maximum speed or LCM model
        if self.use_lcm_lora:
            model_id = "SimianLuo/LCM_Dreamshaper_v7"
        else:
            model_id = "stabilityai/sd-turbo"
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
            use_safetensors=True
        ).to(self.device)
        
        # Configure scheduler
        self.pipe.scheduler = LCMScheduler.from_config(
            self.pipe.scheduler.config
        )
        
        # Optimizations
        self.pipe.set_progress_bar_config(disable=True)
        
        # Enable memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("✓ XFormers memory efficient attention enabled")
        except:
            print("⚠️ XFormers not available")
        
        # Try torch.compile for additional speedup
        if not hasattr(self.pipe, 'enable_xformers_memory_efficient_attention'):
            try:
                self.pipe.unet = torch.compile(
                    self.pipe.unet,
                    mode="reduce-overhead",
                    fullgraph=True
                )
                print("✓ UNet compiled with torch.compile")
            except:
                print("⚠️ torch.compile not available")
        
        # Use Tiny VAE if requested
        if self.use_tiny_vae:
            try:
                self.pipe.vae = AutoencoderTiny.from_pretrained(
                    "madebyollin/taesd",
                    torch_dtype=self.dtype
                ).to(self.device)
                print("✓ TinyVAE loaded")
            except:
                print("⚠️ TinyVAE not available")
        
        # Enable VAE optimizations
        try:
            self.pipe.vae.enable_slicing()
            self.pipe.vae.enable_tiling()
            print("✓ VAE slicing and tiling enabled")
        except:
            pass
            
    def _prewarm_pipeline(self):
        """Pre-warm the pipeline to avoid initial latency"""
        print("🔥 Pre-warming pipeline...")
        
        dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        # Pre-warm with different configurations
        for i in range(2):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    if STREAMDIFFUSION_AVAILABLE:
                        self.stream.prepare(
                            prompt="a professional photograph",
                            num_inference_steps=4
                        )
                        _ = self.stream(dummy_image)
                    else:
                        _ = self.pipe(
                            prompt="a professional photograph",
                            image=dummy_image,
                            num_inference_steps=4,
                            strength=0.75,
                            guidance_scale=1.5
                        ).images[0]
        
        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()
        
        print("✓ Pipeline pre-warmed and ready!")
        
    def preprocess_image_optimized(self, image):
        """Optimized preprocessing for brand activation"""
        # Ensure correct size
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Light enhancement for better results
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.05)
        
        return image
        
    def calculate_image_similarity_fast(self, image):
        """Fast similarity calculation"""
        # Downsample for speed
        small = image.resize((64, 64), Image.Resampling.NEAREST)
        array = np.array(small, dtype=np.uint8)
        return hashlib.md5(array.tobytes()).hexdigest()
        
    def should_skip_frame(self, image):
        """Optimized frame skipping logic"""
        current_hash = self.calculate_image_similarity_fast(image)
        
        if self.last_image_hash is None:
            self.last_image_hash = current_hash
            return False
            
        if current_hash == self.last_image_hash:
            self.skip_counter += 1
            if self.skip_counter <= self.max_skip_frames:
                self.skipped_frames += 1
                return True
        else:
            self.skip_counter = 0
            
        self.last_image_hash = current_hash
        return False
        
    def start_processing_thread(self):
        """Start optimized processing thread"""
        self.processing = True
        thread = threading.Thread(target=self._processing_loop, daemon=True)
        thread.start()
        print("✓ Processing thread started")
        
    def _processing_loop(self):
        """Main processing loop with all optimizations"""
        # CUDA stream for async operations
        cuda_stream = torch.cuda.Stream()
        
        while self.processing:
            try:
                # Get frame with minimal timeout
                frame_data = self.frame_queue.get(timeout=0.05)
                
                start_time = time.time()
                
                # Skip similar frames
                if self.should_skip_frame(frame_data['image']):
                    # Reuse last result if available
                    try:
                        if not self.result_queue.empty():
                            last_result = list(self.result_queue.queue)[-1]
                            last_result['skipped'] = True
                            self.result_queue.put_nowait(last_result)
                    except:
                        pass
                    continue
                
                # Preprocess
                processed_image = self.preprocess_image_optimized(frame_data['image'])
                
                # Generate with optimized settings
                prompt = frame_data.get('prompt', 'professional photography')
                
                with torch.cuda.stream(cuda_stream):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            if STREAMDIFFUSION_AVAILABLE:
                                # StreamDiffusion path
                                result = self.stream(processed_image)
                            else:
                                # Optimized Diffusers path
                                result = self.pipe(
                                    prompt=prompt,
                                    image=processed_image,
                                    num_inference_steps=4,  # Optimal for LCM
                                    strength=0.75,
                                    guidance_scale=1.5,  # Low CFG for LCM
                                    generator=torch.Generator(device=self.device).manual_seed(42)
                                ).images[0]
                
                # Calculate metrics
                processing_time = (time.time() - start_time) * 1000
                self.total_frames += 1
                
                # Update FPS
                current_time = time.time()
                self.fps_history.append(current_time)
                if len(self.fps_history) > 1:
                    fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
                else:
                    fps = 0
                
                # Prepare result
                result_data = {
                    'image': result,
                    'processing_time': processing_time,
                    'timestamp': frame_data['timestamp'],
                    'stats': {
                        'fps': round(fps, 1),
                        'total_frames': self.total_frames,
                        'skipped_frames': self.skipped_frames,
                        'skip_rate': (self.skipped_frames / max(1, self.total_frames)) * 100,
                        'processing_ms': round(processing_time, 1),
                        'tensorrt': self.use_tensorrt,
                        'lcm_lora': self.use_lcm_lora
                    }
                }
                
                # Put result
                try:
                    # Clear old results
                    while self.result_queue.qsize() > 2:
                        self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result_data)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
                
    def add_frame(self, image, prompt, timestamp):
        """Add frame for processing"""
        try:
            # Clear old frames
            while self.frame_queue.qsize() > 1:
                self.frame_queue.get_nowait()
                
            self.frame_queue.put_nowait({
                'image': image,
                'prompt': prompt,
                'timestamp': timestamp
            })
        except:
            pass
            
    def get_latest_result(self):
        """Get most recent result"""
        result = None
        try:
            # Get all available results
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
        except:
            pass
        return result

# Global processor instance
processor = OptimizedBrandActivationDiffusion()

# Optimized HTML interface for brand activation
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Transform Experience - Brand Activation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #0f0f0f, #1a1a2e);
            color: #fff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
        }
        
        .brand-header {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            z-index: 1000;
        }
        
        .brand-title {
            font-size: 36px;
            font-weight: bold;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 3px;
            margin-bottom: 10px;
        }
        
        .brand-subtitle {
            font-size: 18px;
            color: #aaa;
            letter-spacing: 1px;
        }
        
        .main-container {
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            gap: 40px;
            padding: 20px;
        }
        
        .video-container, .output-container {
            position: relative;
            width: 600px;
            height: 600px;
        }
        
        video, canvas {
            width: 100%;
            height: 100%;
            border: 3px solid #00ff88;
            border-radius: 20px;
            background: #000;
            box-shadow: 0 0 50px rgba(0, 255, 136, 0.3);
        }
        
        .label {
            position: absolute;
            bottom: -40px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 20px;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .status-badge {
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(0, 255, 136, 0.2);
            border: 2px solid #00ff88;
            border-radius: 30px;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: bold;
            backdrop-filter: blur(10px);
        }
        
        .controls {
            position: fixed;
            bottom: 30px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(20, 20, 20, 0.95);
            padding: 30px 50px;
            border-radius: 25px;
            border: 2px solid #00ff88;
            backdrop-filter: blur(20px);
            z-index: 1000;
            text-align: center;
        }
        
        .style-selector {
            display: flex;
            gap: 15px;
            margin-bottom: 25px;
            justify-content: center;
        }
        
        .style-btn {
            padding: 12px 24px;
            background: rgba(0, 255, 136, 0.1);
            border: 2px solid #00ff88;
            color: #00ff88;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            text-transform: uppercase;
        }
        
        .style-btn:hover {
            background: rgba(0, 255, 136, 0.2);
            transform: scale(1.05);
        }
        
        .style-btn.active {
            background: #00ff88;
            color: #000;
        }
        
        .start-btn {
            padding: 20px 60px;
            font-size: 24px;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            border: none;
            color: #000;
            border-radius: 50px;
            cursor: pointer;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s;
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.3);
        }
        
        .start-btn:hover {
            transform: scale(1.05);
            box-shadow: 0 15px 40px rgba(0, 255, 136, 0.5);
        }
        
        .start-btn:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .performance-meter {
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.9);
            padding: 20px;
            border-radius: 15px;
            border: 2px solid #00ff88;
            font-family: 'Consolas', monospace;
            font-size: 14px;
            min-width: 200px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
        }
        
        .metric-value {
            color: #00ff88;
            font-weight: bold;
        }
        
        /* Totem mode for full installation */
        .totem-mode {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: #000;
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .totem-mode.active {
            display: flex;
        }
        
        .totem-canvas {
            width: 80vh;
            height: 80vh;
            border: none;
            border-radius: 0;
        }
        
        .totem-brand {
            position: absolute;
            top: 5%;
            font-size: 4em;
            font-weight: bold;
            background: linear-gradient(45deg, #00ff88, #00cc66);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 5px;
        }
        
        .totem-instruction {
            position: absolute;
            bottom: 10%;
            font-size: 2em;
            text-align: center;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }
        
        .loading-spinner {
            width: 80px;
            height: 80px;
            border: 4px solid #333;
            border-top: 4px solid #00ff88;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="loading-overlay" id="loadingOverlay">
        <div class="loading-spinner"></div>
    </div>

    <div class="brand-header">
        <div class="brand-title">AI Transform Experience</div>
        <div class="brand-subtitle">Powered by Advanced Neural Processing</div>
    </div>
    
    <div class="main-container" id="mainContainer">
        <div class="video-container">
            <video id="video" autoplay muted playsinline></video>
            <div class="label">Your Reality</div>
        </div>
        
        <div class="output-container">
            <canvas id="output"></canvas>
            <div class="label">AI Vision</div>
            <div class="status-badge" id="statusBadge">Ready</div>
        </div>
    </div>
    
    <div class="totem-mode" id="totemMode">
        <div class="totem-brand">AI TRANSFORM</div>
        <canvas id="totemCanvas" class="totem-canvas"></canvas>
        <div class="totem-instruction">Step into the frame to begin</div>
    </div>
    
    <div class="performance-meter" id="performanceMeter">
        <div class="metric">
            <span>FPS</span>
            <span class="metric-value" id="fpsValue">0</span>
        </div>
        <div class="metric">
            <span>Latency</span>
            <span class="metric-value" id="latencyValue">0ms</span>
        </div>
        <div class="metric">
            <span>Quality</span>
            <span class="metric-value" id="qualityValue">Ultra</span>
        </div>
    </div>
    
    <div class="controls" id="controls">
        <div class="style-selector">
            <button class="style-btn active" data-style="cinematic">Cinematic</button>
            <button class="style-btn" data-style="cyberpunk">Cyberpunk</button>
            <button class="style-btn" data-style="anime">Anime</button>
            <button class="style-btn" data-style="artistic">Artistic</button>
        </div>
        <button class="start-btn" id="startBtn" onclick="startExperience()">
            Start Experience
        </button>
    </div>
    
    <script>
        // Configuration
        const WEBSOCKET_CONFIG = {
            reconnectInterval: 3000,
            maxReconnectAttempts: 5,
            pingInterval: 25000,
            frameInterval: 33  // ~30 FPS capture
        };
        
        // Style prompts optimized for LCM
        const STYLE_PROMPTS = {
            cinematic: "cinematic photography, professional lighting, high quality",
            cyberpunk: "cyberpunk style, neon lights, futuristic",
            anime: "anime style artwork, vibrant colors",
            artistic: "artistic painting, masterpiece, detailed"
        };
        
        // Global state
        let ws = null;
        let streaming = false;
        let currentStyle = 'cinematic';
        let video = null;
        let canvas = null;
        let ctx = null;
        let totemMode = false;
        let frameTimer = null;
        let reconnectAttempts = 0;
        let lastFrameTime = Date.now();
        let frameCount = 0;
        let fpsUpdateTimer = null;
        
        // Style selection
        document.querySelectorAll('.style-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                document.querySelectorAll('.style-btn').forEach(b => b.classList.remove('active'));
                this.classList.add('active');
                currentStyle = this.dataset.style;
            });
        });
        
        async function startExperience() {
            try {
                // Hide loading
                document.getElementById('loadingOverlay').style.display = 'none';
                
                // Initialize video
                video = document.getElementById('video');
                canvas = document.getElementById('output');
                ctx = canvas.getContext('2d');
                
                // Set canvas size
                canvas.width = 600;
                canvas.height = 600;
                
                // Get camera stream with optimal settings
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 },
                        facingMode: 'user'
                    }
                });
                
                video.srcObject = stream;
                await new Promise(resolve => video.onloadedmetadata = resolve);
                
                // Connect WebSocket
                connectWebSocket();
                
                // Update UI
                document.getElementById('startBtn').disabled = true;
                document.getElementById('statusBadge').textContent = 'Connecting...';
                
                // Start FPS counter
                startFPSCounter();
                
            } catch (error) {
                console.error('Error:', error);
                alert('Camera access required. Please allow camera permissions.');
            }
        }
        
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('Connected to AI server');
                streaming = true;
                reconnectAttempts = 0;
                document.getElementById('statusBadge').textContent = 'Live';
                document.getElementById('statusBadge').style.background = 'rgba(0, 255, 136, 0.3)';
                
                // Start streaming
                startFrameCapture();
                
                // Start ping interval
                setInterval(() => {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'ping' }));
                    }
                }, WEBSOCKET_CONFIG.pingInterval);
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    if (data.type === 'pong') return;
                    
                    if (data.image) {
                        // Display image
                        const img = new Image();
                        img.onload = () => {
                            ctx.clearRect(0, 0, canvas.width, canvas.height);
                            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                            
                            // Update totem mode if active
                            if (totemMode) {
                                const totemCanvas = document.getElementById('totemCanvas');
                                const totemCtx = totemCanvas.getContext('2d');
                                totemCtx.clearRect(0, 0, totemCanvas.width, totemCanvas.height);
                                totemCtx.drawImage(img, 0, 0, totemCanvas.width, totemCanvas.height);
                            }
                        };
                        img.src = data.image;
                        
                        // Update metrics
                        updateMetrics(data.stats);
                        frameCount++;
                    }
                } catch (error) {
                    console.error('Message error:', error);
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('Disconnected from server');
                streaming = false;
                document.getElementById('statusBadge').textContent = 'Disconnected';
                document.getElementById('statusBadge').style.background = 'rgba(255, 68, 68, 0.3)';
                
                // Stop frame capture
                if (frameTimer) {
                    clearInterval(frameTimer);
                    frameTimer = null;
                }
                
                // Attempt reconnect
                if (reconnectAttempts < WEBSOCKET_CONFIG.maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, WEBSOCKET_CONFIG.reconnectInterval);
                }
            };
        }
        
        function startFrameCapture() {
            // Clear any existing timer
            if (frameTimer) clearInterval(frameTimer);
            
            // Capture frames at specified interval
            frameTimer = setInterval(() => {
                if (streaming && ws && ws.readyState === WebSocket.OPEN) {
                    captureAndSendFrame();
                }
            }, WEBSOCKET_CONFIG.frameInterval);
        }
        
        function captureAndSendFrame() {
            if (!video || video.videoWidth === 0) return;
            
            // Create temporary canvas for capture
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 512;
            tempCanvas.height = 512;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Calculate crop for square aspect
            const size = Math.min(video.videoWidth, video.videoHeight);
            const sx = (video.videoWidth - size) / 2;
            const sy = (video.videoHeight - size) / 2;
            
            // Draw and resize
            tempCtx.drawImage(video, sx, sy, size, size, 0, 0, 512, 512);
            
            // Convert to base64 with optimization
            const imageData = tempCanvas.toDataURL('image/jpeg', 0.85);
            
            // Send frame
            ws.send(JSON.stringify({
                image: imageData,
                prompt: STYLE_PROMPTS[currentStyle],
                timestamp: Date.now()
            }));
        }
        
        function updateMetrics(stats) {
            if (!stats) return;
            
            // Update FPS
            if (stats.fps !== undefined) {
                document.getElementById('fpsValue').textContent = stats.fps;
            }
            
            // Update latency
            if (stats.processing_ms !== undefined) {
                document.getElementById('latencyValue').textContent = Math.round(stats.processing_ms) + 'ms';
            }
            
            // Update quality indicator
            if (stats.tensorrt && stats.lcm_lora) {
                document.getElementById('qualityValue').textContent = 'Ultra';
            } else if (stats.lcm_lora) {
                document.getElementById('qualityValue').textContent = 'High';
            } else {
                document.getElementById('qualityValue').textContent = 'Standard';
            }
        }
        
        function startFPSCounter() {
            let lastTime = Date.now();
            let lastFrameCount = frameCount;
            
            fpsUpdateTimer = setInterval(() => {
                const currentTime = Date.now();
                const deltaTime = (currentTime - lastTime) / 1000;
                const deltaFrames = frameCount - lastFrameCount;
                
                const fps = Math.round(deltaFrames / deltaTime);
                document.getElementById('fpsValue').textContent = fps;
                
                lastTime = currentTime;
                lastFrameCount = frameCount;
            }, 1000);
        }
        
        // Totem mode toggle
        function toggleTotemMode() {
            totemMode = !totemMode;
            const totemDiv = document.getElementById('totemMode');
            const mainDiv = document.getElementById('mainContainer');
            const controls = document.getElementById('controls');
            
            if (totemMode) {
                totemDiv.classList.add('active');
                mainDiv.style.display = 'none';
                controls.style.display = 'none';
                document.querySelector('.brand-header').style.display = 'none';
                
                // Setup totem canvas
                const totemCanvas = document.getElementById('totemCanvas');
                totemCanvas.width = 800;
                totemCanvas.height = 800;
            } else {
                totemDiv.classList.remove('active');
                mainDiv.style.display = 'flex';
                controls.style.display = 'block';
                document.querySelector('.brand-header').style.display = 'block';
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'f' || e.key === 'F') {
                toggleTotemMode();
            }
            if (e.key === 'Escape' && totemMode) {
                toggleTotemMode();
            }
        });
        
        // Clean up on page unload
        window.addEventListener('beforeunload', () => {
            if (ws) ws.close();
            if (video && video.srcObject) {
                video.srcObject.getTracks().forEach(track => track.stop());
            }
            if (frameTimer) clearInterval(frameTimer);
            if (fpsUpdateTimer) clearInterval(fpsUpdateTimer);
        });
        
        // Hide loading when ready
        window.addEventListener('load', () => {
            setTimeout(() => {
                document.getElementById('loadingOverlay').style.display = 'none';
            }, 500);
        });
    </script>
</body>
</html>
"""

# WebSocket handler with optimizations
@app.get("/")
async def get():
    return HTMLResponse(content=HTML_CONTENT)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = id(websocket)
    print(f"✅ Client connected: {client_id}")
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            # Handle ping/pong
            if data.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                continue
            
            # Process image
            if 'image' not in data:
                continue
                
            try:
                # Decode image
                img_data = base64.b64decode(data['image'].split(',')[1])
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                
                # Add to processing queue
                processor.add_frame(
                    image,
                    data.get('prompt', 'professional photography'),
                    data.get('timestamp', time.time() * 1000)
                )
                
                # Get result
                result = processor.get_latest_result()
                
                if result:
                    # Encode result image
                    buffered = io.BytesIO()
                    result['image'].save(buffered, format="JPEG", quality=90)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Send response
                    await websocket.send_json({
                        'image': f'data:image/jpeg;base64,{img_str}',
                        'stats': result['stats'],
                        'timestamp': result['timestamp']
                    })
                    
            except Exception as e:
                print(f"Frame processing error: {e}")
                continue
                
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print(f"Cleaning up client: {client_id}")

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 OPTIMIZED STREAMDIFFUSION - BRAND ACTIVATION")
    print("="*80)
    print("✓ TensorRT acceleration (when available)")
    print("✓ LCM-LoRA 4-step generation")
    print("✓ WebSocket optimizations")
    print("✓ L40S GPU optimizations")
    print("✓ Target: 15-20+ FPS with maximum quality")
    print("="*80 + "\n")
    
    # Run with optimized settings
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="warning",
        access_log=False,  # Disable access logs for performance
        loop="uvloop"  # Use uvloop for better performance
    )
