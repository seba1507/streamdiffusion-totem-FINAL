# RunPod Deployment Guide - Enhanced StreamDiffusion

## 🚀 Deployment Steps with RunPod MCP

### 1. Create Pod using MCP

Ask Claude with RunPod MCP:
```
Create a GPU pod with these specs:
- GPU: RTX 4090 24GB (or RTX 3090)
- Template: RunPod PyTorch 2.1
- Container Disk: 50GB
- Expose port 8000 as HTTP
- Name: enhanced-diffusion-totem
```

### 2. Initial Setup Commands

Once the pod is created, execute these commands in sequence:

```bash
# Navigate to workspace
cd /workspace

# Create project directory
mkdir enhanced-diffusion && cd enhanced-diffusion

# Download files directly (Alternative 1: Using wget if you host them)
# wget https://your-repo/install.sh
# wget https://your-repo/requirements.txt
# wget https://your-repo/server_dotsimulate_enhanced.py

# Or create files manually (Alternative 2: Copy-paste)
# Use the RunPod file manager or terminal editor
```

### 3. Quick Setup Script

Create this all-in-one setup script:

```bash
cat > quick_setup.sh << 'EOSETUP'
#!/bin/bash

echo "🚀 Quick Setup - Enhanced StreamDiffusion"

# Create all necessary files
echo "📝 Creating files..."

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Core PyTorch with CUDA 11.8
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.1.0
torchvision==0.16.0

# Diffusers and related
diffusers==0.24.0
transformers==4.36.2
accelerate==0.25.0
huggingface_hub==0.19.4
xformers==0.0.22

# Web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0

# Image processing
pillow==10.1.0
opencv-python-headless==4.8.1.78
numpy==1.24.3

# Additional utilities
scipy==1.11.4
safetensors==0.4.1
EOF

# Download the main server script
echo "📥 Downloading server script..."
# NOTE: Replace this URL with your actual file location
curl -o server_dotsimulate_enhanced.py "YOUR_SCRIPT_URL_HERE"

# Make sure the file exists
if [ ! -f "server_dotsimulate_enhanced.py" ]; then
    echo "❌ Error: server_dotsimulate_enhanced.py not found!"
    echo "Please upload the file manually"
    exit 1
fi

# Run installation
echo "📦 Installing dependencies..."
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete! Run with: python3 server_dotsimulate_enhanced.py"
EOSETUP

chmod +x quick_setup.sh
./quick_setup.sh
```

### 4. Manual File Creation

If you need to create `server_dotsimulate_enhanced.py` manually:

```bash
# Open nano editor
nano server_dotsimulate_enhanced.py

# Paste the entire script content
# Save with Ctrl+X, Y, Enter
```

### 5. Verify Installation

```bash
# Test CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test imports
python3 -c "import diffusers, fastapi, PIL; print('✅ All imports OK')"
```

### 6. Start the Server

```bash
# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Run server
python3 server_dotsimulate_enhanced.py
```

### 7. Access the Application

Your application will be available at:
```
https://[YOUR-POD-ID]-8000.proxy.runpod.net/
```

Find your Pod ID in the RunPod dashboard or ask the MCP:
```
Show me the details of my enhanced-diffusion-totem pod
```

## 🔧 Troubleshooting Commands

```bash
# Check GPU
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check Python packages
pip list | grep -E "torch|diffusers|fastapi"

# Test server without running
python3 -c "import server_dotsimulate_enhanced"

# Check port
netstat -tlnp | grep 8000
```

## 📊 Performance Optimization

For RunPod specifically:

```bash
# Set performance mode
nvidia-smi -pm 1

# Check current clock speeds
nvidia-smi -q -d CLOCK

# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader -l 1
```

## 🛡️ Security Notes

1. The application exposes port 8000 publicly through RunPod proxy
2. No authentication is implemented by default
3. Consider adding basic auth for production use
4. Monitor usage to control costs

## 💰 Cost Optimization

- Use pod stop/start to pause when not in use
- Set up auto-stop after inactivity
- Monitor GPU hours consumed
- Consider using spot instances for testing

## 📱 Testing the Application

1. Open the RunPod proxy URL in browser
2. Allow camera permissions
3. Click "Start Enhanced Stream"
4. Adjust parameters:
   - Strength: 0.75 (default)
   - CFG Scale: 8.0 (default)
   - Custom prompts

## 🎯 Quick Test URLs

Once running, test these endpoints:
- Main app: `https://[POD-ID]-8000.proxy.runpod.net/`
- Health check: `https://[POD-ID]-8000.proxy.runpod.net/docs`

## 📝 Notes

- First run will download the model (~2GB)
- Initial frame generation may be slow (pre-warming)
- Optimal performance after 10-20 frames
- Memory usage: ~8GB VRAM typical