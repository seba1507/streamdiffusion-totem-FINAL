#!/bin/bash

echo "===================================================================="
echo "🎨 ENHANCED STREAMDIFFUSION - DOTSIMULATE INSTALLATION"
echo "===================================================================="

# Set environment variables for CUDA
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Update system
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y python3-pip python3-dev git wget

# Verify CUDA
echo "🔍 Verifying CUDA installation..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: CUDA not found. Please ensure you're using a GPU pod."
    exit 1
fi

echo "✅ CUDA detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create virtual environment (optional, but recommended)
echo "🐍 Setting up Python environment..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8 first
echo "🔥 Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Verify PyTorch CUDA
echo "🔍 Verifying PyTorch CUDA support..."
python3 -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"

# Install remaining requirements
echo "📦 Installing remaining dependencies..."
pip install -r requirements.txt

# Pre-download the model to cache
echo "📥 Pre-downloading LCM Dreamshaper v7 model..."
python3 -c "
from diffusers import AutoPipelineForImage2Image
import torch

print('Downloading model...')
try:
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
"

# Create run script
echo "📝 Creating run script..."
cat > run.sh << 'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "🚀 Starting Enhanced StreamDiffusion Server..."
python3 server_dotsimulate_enhanced.py
EOF

chmod +x run.sh

# Create test script
echo "📝 Creating test script..."
cat > test_cuda.py << 'EOF'
import torch
import sys

print("=" * 60)
print("CUDA Test")
print("=" * 60)

if torch.cuda.is_available():
    print(f"✅ CUDA is available")
    print(f"📍 Device: {torch.cuda.get_device_name(0)}")
    print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🔢 CUDA Version: {torch.version.cuda}")
    
    # Test tensor operation
    x = torch.randn(1, 3, 512, 512).cuda()
    print(f"✅ Test tensor created on GPU: {x.shape}")
else:
    print("❌ CUDA is NOT available")
    sys.exit(1)
EOF

# Run CUDA test
echo "🧪 Running CUDA test..."
python3 test_cuda.py

echo ""
echo "===================================================================="
echo "✅ Installation complete!"
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
echo "  http://0.0.0.0:8000"
echo ""
echo "===================================================================="
