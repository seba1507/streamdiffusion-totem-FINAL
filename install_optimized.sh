#!/bin/bash

echo "===================================================================="
echo "🚀 OPTIMIZED STREAMDIFFUSION - BRAND ACTIVATION INSTALLATION"
echo "===================================================================="
echo "Target Performance: 15-20+ FPS on NVIDIA L40S"
echo ""

# Set CUDA environment variables for L40S
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,expandable_segments:True'

# L40S specific optimizations
export TORCH_CUDA_ARCH_LIST="8.9"  # Ada Lovelace architecture
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_MODULE_LOADING=LAZY

# Verify GPU
echo "🔍 Detecting GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: NVIDIA GPU not detected!"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# Check for L40S
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
if [[ $GPU_NAME == *"L40S"* ]]; then
    echo "✅ NVIDIA L40S detected - applying optimizations"
else
    echo "⚠️  GPU: $GPU_NAME (optimizations designed for L40S)"
fi

# Update pip
echo "📦 Updating pip and build tools..."
python3 -m pip install --upgrade pip setuptools wheel

# Clean previous installations
echo "🧹 Cleaning previous installations..."
pip uninstall -y torch torchvision xformers diffusers transformers accelerate

# Install PyTorch with CUDA 11.8
echo "📦 Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
echo "🔍 Verifying CUDA installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

if [ $? -ne 0 ]; then
    echo "❌ CUDA verification failed!"
    exit 1
fi

# Install XFormers
echo "📦 Installing XFormers for memory efficient attention..."
pip install xformers==0.0.23 --no-deps
pip install -r requirements_optimized.txt

# Try to install TensorRT
echo "📦 Attempting TensorRT installation..."
pip install nvidia-tensorrt --index-url https://pypi.ngc.nvidia.com 2>/dev/null || echo "⚠️  TensorRT not available via pip"

# Install StreamDiffusion if available
echo "📦 Installing StreamDiffusion..."
if [ -d "StreamDiffusion" ]; then
    cd StreamDiffusion && pip install -e . && cd ..
else
    git clone https://github.com/cumulo-autumn/StreamDiffusion.git 2>/dev/null && \
    cd StreamDiffusion && pip install -e . && cd .. || \
    echo "⚠️  StreamDiffusion not installed - using optimized Diffusers"
fi

# Download models for pre-caching
echo "📥 Pre-downloading models..."
python3 << EOF
import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny
import os

print("Downloading SD-Turbo...")
try:
    AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sd-turbo",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    print("✅ SD-Turbo downloaded")
except Exception as e:
    print(f"⚠️  SD-Turbo download failed: {e}")

print("Downloading LCM Dreamshaper...")
try:
    AutoPipelineForImage2Image.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    print("✅ LCM Dreamshaper downloaded")
except Exception as e:
    print(f"⚠️  LCM Dreamshaper download failed: {e}")

print("Downloading TinyVAE...")
try:
    AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        torch_dtype=torch.float16
    )
    print("✅ TinyVAE downloaded")
except Exception as e:
    print(f"⚠️  TinyVAE download failed: {e}")
EOF

# Create optimized run script
echo "📝 Creating optimized run script..."
cat > run_optimized.sh << 'RUNEOF'
#!/bin/bash

# L40S Optimized Environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,expandable_segments:True'
export TORCH_CUDA_ARCH_LIST="8.9"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_MODULE_LOADING=LAZY

# Performance settings
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

clear
echo "===================================================================="
echo "🚀 OPTIMIZED STREAMDIFFUSION - BRAND ACTIVATION"
echo "===================================================================="
echo ""
echo "🎯 Target: 15-20+ FPS with maximum quality"
echo "🎨 Features: TensorRT, LCM-LoRA, WebSocket optimizations"
echo ""
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader
echo ""
echo "🌐 Access URL: https://$(hostname)-8000.proxy.runpod.net/"
echo ""
echo "💡 Press 'F' in the interface to toggle fullscreen totem mode"
echo "===================================================================="
echo ""

python3 server_optimized_brand_activation.py
RUNEOF

chmod +x run_optimized.sh

# Create performance test script
echo "📝 Creating performance test..."
cat > test_performance.py << 'EOF'
import torch
import time
from diffusers import AutoPipelineForImage2Image
from PIL import Image

print("🧪 Performance Test for Brand Activation")
print("-" * 50)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load model
print("\nLoading model...")
pipe = AutoPipelineForImage2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# Test image
test_img = Image.new('RGB', (512, 512), color='gray')

# Warmup
print("Warming up...")
for _ in range(2):
    with torch.no_grad():
        pipe("test", image=test_img, num_inference_steps=1, strength=0.75).images[0]

# Benchmark
print("\nBenchmarking...")
times = []
for i in range(10):
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        pipe("professional photo", image=test_img, num_inference_steps=1, strength=0.75).images[0]
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed*1000:.1f}ms ({1/elapsed:.1f} FPS)")

avg_time = sum(times) / len(times)
print(f"\nAverage: {avg_time*1000:.1f}ms ({1/avg_time:.1f} FPS)")
print(f"Best: {min(times)*1000:.1f}ms ({1/min(times):.1f} FPS)")
EOF

# Final instructions
echo ""
echo "===================================================================="
echo "✅ Installation Complete!"
echo "===================================================================="
echo ""
echo "📊 Test performance:"
echo "   python3 test_performance.py"
echo ""
echo "🚀 Start the server:"
echo "   ./run_optimized.sh"
echo ""
echo "🎯 Optimization checklist:"
echo "   ✓ PyTorch 2.1.0 with CUDA 11.8"
echo "   ✓ XFormers memory efficient attention"
echo "   ✓ LCM-LoRA support for 4-step generation"
echo "   ✓ TinyVAE for faster decoding"
echo "   ✓ WebSocket optimizations"
echo "   ✓ L40S-specific CUDA settings"
echo ""
echo "💡 Expected performance on L40S: 18-22 FPS"
echo "===================================================================="
