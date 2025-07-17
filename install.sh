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
nvidia-smi --query-gpu=name,memory.total --format=csv

# Clean previous installations to avoid conflicts
echo "🧹 Cleaning previous installations..."
pip uninstall torch torchvision xformers -y 2>/dev/null || true

# Install PyTorch with CUDA 11.8 first
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA
echo "🔍 Verifying PyTorch CUDA support..."
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✅ PyTorch CUDA OK: {torch.cuda.get_device_name(0)}')"

# Install xformers separately (compatible version)
echo "📦 Installing XFormers..."
pip install xformers==0.0.23

# Install remaining requirements
echo "📦 Installing remaining dependencies..."
pip install diffusers==0.24.0 transformers==4.36.2 accelerate==0.25.0 huggingface_hub==0.19.4 safetensors==0.4.1
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0 websockets==12.0
pip install pillow==10.1.0 opencv-python-headless==4.8.1.78 numpy==1.24.3 scipy==1.11.4

# Pre-download the model to cache
echo "📥 Pre-downloading LCM Dreamshaper v7 model..."
python3 << EOF
try:
    from diffusers import AutoPipelineForImage2Image
    import torch
    
    print('Downloading model... This may take a few minutes...')
    pipe = AutoPipelineForImage2Image.from_pretrained(
        'SimianLuo/LCM_Dreamshaper_v7',
        torch_dtype=torch.float16,
        safety_checker=None,
        use_safetensors=True
    )
    print('✅ Model downloaded successfully!')
except Exception as e:
    print(f'⚠️ Model download failed: {e}')
    print('Model will be downloaded on first run.')
EOF

# Create run script if it doesn't exist
if [ ! -f "run.sh" ]; then
    echo "📝 Creating run script..."
    cat > run.sh << 'RUNEOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "🚀 Starting Enhanced StreamDiffusion Server..."
echo "===================================================================="
echo "Server will be available at:"
echo "  Local: http://0.0.0.0:8000"
echo "  RunPod: https://$(hostname)-8000.proxy.runpod.net/"
echo "===================================================================="
python3 server_dotsimulate_enhanced.py
RUNEOF
    chmod +x run.sh
fi

# Create test script
echo "📝 Creating test script..."
cat > test_installation.py << 'EOF'
import sys
print("=" * 60)
print("🧪 Testing Enhanced StreamDiffusion Installation")
print("=" * 60)

# Test CUDA
try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA NOT available")
        sys.exit(1)
except Exception as e:
    print(f"❌ PyTorch error: {e}")
    sys.exit(1)

# Test all imports
modules = {
    "diffusers": "Diffusers",
    "transformers": "Transformers", 
    "xformers": "XFormers",
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "cv2": "OpenCV",
    "PIL": "Pillow",
    "numpy": "NumPy"
}

all_ok = True
for module, name in modules.items():
    try:
        __import__(module)
        print(f"✅ {name} imported successfully")
    except ImportError as e:
        print(f"❌ {name} import failed: {e}")
        all_ok = False

print("=" * 60)
if all_ok:
    print("✅ All tests passed! Ready to run server.")
else:
    print("❌ Some imports failed. Check errors above.")
    sys.exit(1)
EOF

# Run installation test
echo "🧪 Running installation test..."
python3 test_installation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "✅ Installation complete and verified!"
    echo "===================================================================="
    echo ""
    echo "To start the server:"
    echo "  ./run.sh"
    echo ""
    echo "Or manually:"
    echo "  export CUDA_VISIBLE_DEVICES=0"
    echo "  python3 server_dotsimulate_enhanced.py"
    echo ""
    echo "The server will be available at:"
    echo "  https://$(hostname)-8000.proxy.runpod.net/"
    echo ""
    echo "===================================================================="
else
    echo ""
    echo "❌ Installation test failed. Please check the errors above."
    exit 1
fi
