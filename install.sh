#!/bin/bash

echo "===================================================================="
echo "🎨 ENHANCED STREAMDIFFUSION - DOTSIMULATE INSTALLATION"
echo "===================================================================="

# Set environment variables for CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Verify we're in the right directory
if [ ! -f "server_dotsimulate_enhanced.py" ]; then
    echo "❌ ERROR: server_dotsimulate_enhanced.py not found!"
    echo "Make sure you're in the correct directory"
    exit 1
fi

# Check for requirements.txt
if [ ! -f "requirements.txt" ]; then
    echo "❌ ERROR: requirements.txt not found!"
    exit 1
fi

# Update system
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Verify CUDA
echo "🔍 Verifying CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: CUDA not found. Please ensure you're using a GPU pod."
    exit 1
fi

echo "✅ CUDA detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Get GPU name to check compatibility
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
echo "🎮 GPU: $GPU_NAME"

# Clean previous installations to avoid conflicts
echo "🧹 Cleaning previous installations..."
pip uninstall torch torchvision xformers -y 2>/dev/null || true

# Install from requirements.txt
echo "📦 Installing from requirements.txt..."
pip install -r requirements.txt

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
python3 << EOF
import sys
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} installed")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test GPU operations
        x = torch.randn(1, 3, 512, 512).cuda()
        print("✅ GPU operations working correctly")
    else:
        print("❌ CUDA NOT available - GPU will not be used!")
        print("⚠️  The application will run slowly on CPU")
except Exception as e:
    print(f"❌ PyTorch error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo "❌ PyTorch verification failed!"
    exit 1
fi

# Verify all critical imports
echo "🔍 Verifying all dependencies..."
python3 << EOF
import sys
all_ok = True

modules = [
    ("torch", "PyTorch"),
    ("torchvision", "TorchVision"),
    ("xformers", "XFormers"),
    ("diffusers", "Diffusers"),
    ("transformers", "Transformers"),
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("cv2", "OpenCV"),
    ("PIL", "Pillow"),
    ("numpy", "NumPy"),
    ("tqdm", "tqdm")
]

for module, name in modules:
    try:
        __import__(module)
        print(f"✅ {name}")
    except ImportError as e:
        print(f"❌ {name}: {e}")
        all_ok = False

if not all_ok:
    print("\n❌ Some dependencies failed to install!")
    sys.exit(1)
else:
    print("\n✅ All dependencies installed successfully!")
EOF

if [ $? -ne 0 ]; then
    echo "❌ Dependency verification failed!"
    exit 1
fi

# Pre-download the model to cache (optional but recommended)
echo "📥 Pre-downloading LCM Dreamshaper v7 model..."
echo "This may take 2-3 minutes on first run..."
python3 << EOF
try:
    from diffusers import AutoPipelineForImage2Image
    import torch
    
    print('Downloading model...')
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7',
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    )
    print('✅ Model downloaded successfully!')
    del pipe  # Free memory
    torch.cuda.empty_cache()
except Exception as e:
    print(f'⚠️ Model pre-download failed: {e}')
    print('Model will be downloaded when server starts.')
EOF

# Create run.sh if it doesn't exist
if [ ! -f "run.sh" ]; then
    echo "📝 Creating run.sh..."
    cat > run.sh << 'RUNEOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Get hostname for RunPod URL
HOSTNAME=$(hostname)
RUNPOD_URL="https://${HOSTNAME}-8000.proxy.runpod.net/"

clear
echo "===================================================================="
echo "🎨 ENHANCED STREAMDIFFUSION - DOTSIMULATE"
echo "===================================================================="
echo ""
echo "🚀 Starting server..."
echo ""
echo "📡 Access URLs:"
echo "  Local:  http://0.0.0.0:8000"
echo "  RunPod: ${RUNPOD_URL}"
echo ""
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo ""
echo "💡 Tips:"
echo "  - First run will download models (~2GB)"
echo "  - Initial generation may be slow (pre-warming)"
echo "  - Press Ctrl+C to stop the server"
echo ""
echo "===================================================================="
echo ""

python3 server_dotsimulate_enhanced.py
RUNEOF
    chmod +x run.sh
fi

# Final success message
echo ""
echo "===================================================================="
echo "✅ Installation Complete!"
echo "===================================================================="
echo ""
echo "📊 System Info:"
echo "  GPU: $GPU_NAME"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"
echo ""
echo "🚀 To start the server:"
echo "  ./run.sh"
echo ""
echo "📡 The server will be available at:"
echo "  https://$(hostname)-8000.proxy.runpod.net/"
echo ""
echo "===================================================================="

# Create a quick test script
cat > test_gpu.py << 'EOF'
import torch
import time

print("🧪 Quick GPU Test")
print("-" * 40)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Test computation
    x = torch.randn(1, 3, 512, 512).to(device)
    start = time.time()
    y = x * 2 + 1
    torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000
    
    print(f"✅ GPU computation test: {elapsed:.2f}ms")
    print(f"✅ Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
else:
    print("❌ GPU not available!")
EOF

echo ""
echo "💡 Run 'python3 test_gpu.py' to test GPU performance"
echo ""
