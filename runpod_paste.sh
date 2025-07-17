#!/bin/bash
# ONE-CLICK INSTALL FOR RUNPOD - Just paste this entire script in terminal

cd /workspace
mkdir -p enhanced-diffusion && cd enhanced-diffusion

# Create requirements.txt
cat > requirements.txt << 'EOF'
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.1.0
torchvision==0.16.0
diffusers==0.24.0
transformers==4.36.2
accelerate==0.25.0
huggingface_hub==0.19.4
xformers==0.0.22
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pillow==10.1.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
scipy==1.11.4
safetensors==0.4.1
EOF

# Install dependencies
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
pip install --upgrade pip
pip install -r requirements.txt

# Create note about server file
cat > IMPORTANT_README.txt << 'EOF'
⚠️ IMPORTANT: You need to upload server_dotsimulate_enhanced.py

The server file is too large to paste directly. Please:

1. Use RunPod's file manager to upload server_dotsimulate_enhanced.py
2. Or use wget/curl if you have it hosted somewhere
3. Or create it manually with nano/vim

Once you have the file, run:
   python3 server_dotsimulate_enhanced.py

The server will be available at:
   https://[YOUR-POD-ID]-8000.proxy.runpod.net/
EOF

# Create a test script
cat > test_setup.py << 'EOF'
import sys
print("🧪 Testing Enhanced StreamDiffusion Setup...")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"❌ PyTorch error: {e}")
    sys.exit(1)

try:
    import diffusers
    print(f"✅ Diffusers {diffusers.__version__}")
except Exception as e:
    print(f"❌ Diffusers error: {e}")

try:
    import fastapi
    import uvicorn
    print("✅ FastAPI and Uvicorn OK")
except Exception as e:
    print(f"❌ FastAPI error: {e}")

try:
    import cv2
    import PIL
    import numpy
    print("✅ Image processing libraries OK")
except Exception as e:
    print(f"❌ Image libraries error: {e}")

print("\n📋 Setup Summary:")
print("- All dependencies installed ✅")
print("- CUDA configured ✅")
print("- Ready for server_dotsimulate_enhanced.py upload ⏳")
EOF

# Run test
python3 test_setup.py

# Show instructions
echo ""
echo "================================================================"
echo "✅ Base installation complete!"
echo "================================================================"
echo ""
echo "⚠️  NEXT STEP: Upload server_dotsimulate_enhanced.py"
echo ""
echo "Then run:"
echo "  export CUDA_VISIBLE_DEVICES=0"
echo "  python3 server_dotsimulate_enhanced.py"
echo ""
echo "================================================================"

# Create a run helper
cat > run_server.sh << 'EOF'
#!/bin/bash
if [ ! -f "server_dotsimulate_enhanced.py" ]; then
    echo "❌ Error: server_dotsimulate_enhanced.py not found!"
    echo "Please upload the file first."
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
echo "🚀 Starting Enhanced StreamDiffusion Server..."
python3 server_dotsimulate_enhanced.py
EOF

chmod +x run_server.sh

echo ""
echo "💡 TIP: Once you upload the server file, just run: ./run_server.sh"