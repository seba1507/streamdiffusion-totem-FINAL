# Enhanced StreamDiffusion - DotSimulate Style 🎨

Real-time AI video transformation using advanced DotSimulate techniques for interactive installations and brand activations.

## 🚀 Quick Start - RunPod Deployment

### Option 1: Using RunPod MCP (Recommended)

1. **Create Pod with Claude MCP:**
   ```
   Create a GPU pod named "diffusion-totem" with RTX 4090, PyTorch 2.1 template, 50GB disk, expose port 8000/http
   ```

2. **Deploy from Terminal:**
   ```bash
   cd /workspace && \
   git clone https://github.com/seba1507/streamdiffusion-totem-FINAL.git && \
   cd streamdiffusion-totem-FINAL && \
   chmod +x install.sh run.sh && \
   ./install.sh && \
   ./run.sh
   ```

   For private repository, use:
   ```bash
   git clone https://[USERNAME]:[TOKEN]@github.com/seba1507/streamdiffusion-totem-FINAL.git
   ```

### Option 2: Manual RunPod Setup

1. **Create a GPU Pod:**
   - GPU: RTX 4090 (24GB) or RTX 3090 (24GB)
   - Template: RunPod PyTorch 2.1
   - Disk: 50GB
   - Expose Port: 8000/http

2. **Connect and Deploy:**
   ```bash
   ssh root@[YOUR-POD-ID].proxy.runpod.net
   cd /workspace
   git clone [repository-url]
   cd streamdiffusion-totem-FINAL
   chmod +x install.sh run.sh
   ./install.sh
   ./run.sh
   ```

## 🎯 Features

- **Real-time Processing**: 15-30 FPS at 512x512 resolution
- **DotSimulate Techniques**: Advanced preprocessing and temporal smoothing
- **Totem Mode**: Full-screen display for installations
- **Multiple Styles**: Cyberpunk, Anime, Oil Painting, and more
- **Live Parameter Adjustment**: Change settings without restarting

## 📋 System Requirements

- **GPU**: NVIDIA RTX 3090/4090 (24GB VRAM minimum)
- **CUDA**: 11.8 or higher
- **Python**: 3.10.x
- **OS**: Ubuntu 20.04+ (RunPod default)

## 📊 Performance Expectations

| GPU | Resolution | FPS | Latency |
|-----|------------|-----|---------|
| RTX 4090 | 512x512 | 20-30 | 50-100ms |
| RTX 3090 | 512x512 | 15-20 | 80-150ms |

## 🛠️ Manual Installation

If you need to install manually:

```bash
# 1. Install dependencies
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23
pip install -r requirements.txt

# 2. Run server
export CUDA_VISIBLE_DEVICES=0
python3 server_dotsimulate_enhanced.py
```

## 🔧 Troubleshooting

### Common Issues

**1. CUDA Not Available**
```bash
# Check CUDA
nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Fix
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

**2. Module Import Errors**
```bash
# Test installation
python3 test_installation.py

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

**3. XFormers Conflict**
```bash
# Uninstall and reinstall with correct version
pip uninstall xformers torch torchvision -y
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23
```

## 📁 Repository Structure

```
streamdiffusion-totem-FINAL/
├── server_dotsimulate_enhanced.py  # Main server
├── install.sh                      # Installation script
├── run.sh                          # Run script
├── requirements.txt                # Python dependencies
├── deploy_runpod.sh               # RunPod deployment helper
├── test_installation.py           # Installation tester
└── README.md                      # This file
```

## 🎨 Usage Guide

### Starting the Server
1. Run `./run.sh` or `python3 server_dotsimulate_enhanced.py`
2. Access at: `https://[POD-ID]-8000.proxy.runpod.net/`
3. Allow camera permissions
4. Click "Start Enhanced Stream"

### Controls
- **Strength**: 0.50-0.95 (transformation intensity)
- **CFG Scale**: 7.0-15.0 (prompt adherence)
- **Custom Prompt**: Enter any style description
- **Totem Mode**: Full-screen for installations

### Best Practices
- Good lighting is essential
- Position 1-2 meters from camera
- Slower movements produce better results
- Simple backgrounds work best

## 📈 Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# Server logs
# (visible in terminal where server is running)

# Test endpoint
curl https://[POD-ID]-8000.proxy.runpod.net/
```

## 🔐 Security Notes

- Never share GitHub Personal Access Tokens
- The server has no authentication by default
- Consider adding auth for production use
- Monitor RunPod costs

## 💰 Cost Optimization

- **RTX 4090**: $0.69/hour
- **RTX 3090**: $0.46/hour
- Stop pods when not in use
- Use spot instances for testing

## 🤝 Support

For issues:
1. Check the troubleshooting section
2. Run `python3 test_installation.py`
3. Check GPU status with `nvidia-smi`
4. Verify all files are present

## 📝 License

This project uses open-source models and libraries. Please respect their individual licenses.
