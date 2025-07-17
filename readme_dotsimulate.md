# Enhanced StreamDiffusion - DotSimulate Style 🎨

Real-time AI video transformation using advanced DotSimulate techniques for interactive installations and brand activations.

## 🚀 Quick Start with RunPod MCP

### Step 1: Create Pod using RunPod MCP

Using the RunPod MCP in Claude, create a pod with these specifications:

```
GPU: RTX 4090 (24GB) or RTX 3090 (24GB)
Template: RunPod PyTorch 2.1
Disk: 50GB
Ports: 8000/http
```

Example MCP command:
```
Create a GPU pod with RTX 4090, expose port 8000 for HTTP
```

### Step 2: Connect to Pod

Once the pod is running, connect via SSH or use the web terminal.

### Step 3: Clone or Upload Files

Upload these files to your pod:
- `install.sh`
- `requirements.txt`
- `server_dotsimulate_enhanced.py`

Or create them directly:

```bash
# Create project directory
mkdir enhanced-diffusion && cd enhanced-diffusion

# Create the files (copy content from artifacts)
nano install.sh
nano requirements.txt
nano server_dotsimulate_enhanced.py
```

### Step 4: Run Installation

```bash
# Make install script executable
chmod +x install.sh

# Run installation
./install.sh
```

This will:
- ✅ Configure CUDA environment
- ✅ Install PyTorch with CUDA 11.8
- ✅ Install all dependencies
- ✅ Pre-download the AI model
- ✅ Create run scripts
- ✅ Test CUDA availability

### Step 5: Start the Server

```bash
./run.sh
```

Or manually:
```bash
export CUDA_VISIBLE_DEVICES=0
python3 server_dotsimulate_enhanced.py
```

### Step 6: Access the Application

The server will be available at:
- Local: `http://localhost:8000`
- RunPod Proxy: `https://[YOUR-POD-ID]-8000.proxy.runpod.net/`

## 🎯 Features

### Enhanced DotSimulate Techniques
- **Multi-textured preprocessing** with rich color and density variations
- **Temporal feedback system** for smooth transformations
- **Stochastic similarity filter** for optimized performance
- **Advanced prompt engineering** with quality boosters
- **CFG Scale optimization** (7-15 range for best results)

### Performance Optimizations
- **XFormers memory-efficient attention**
- **VAE slicing and tiling**
- **Adaptive timestep scheduling**
- **Frame skip optimization**
- **torch.compile support** (when available)

### User Interface
- **Real-time metrics dashboard**
- **Totem Mode** for full-screen installations
- **Live parameter adjustment**
- **Enhanced visual feedback**

## 📊 Expected Performance

| GPU | Resolution | FPS | Latency |
|-----|------------|-----|---------|
| RTX 4090 | 512x512 | 20-30 | 50-100ms |
| RTX 3090 | 512x512 | 15-20 | 80-150ms |
| RTX 3080 | 512x512 | 10-15 | 100-200ms |

## 🎨 Usage Tips

### For Best Visual Results
1. **Lighting**: Ensure good, even lighting on subjects
2. **Distance**: Position 1-2 meters from camera
3. **Background**: Simple backgrounds work best
4. **Movement**: Slower movements produce better results

### Prompt Engineering
- Start with base style: "cyberpunk", "anime", "oil painting"
- System auto-enhances with quality modifiers
- Adjust strength (0.5-0.95) for transformation intensity
- Use CFG scale (7-15) for prompt adherence

### Totem Mode
- Press "Totem Mode" button for installation display
- ESC key to exit
- F11 shortcut to enter (when streaming)

## 🛠️ Troubleshooting

### CUDA Not Available
```bash
# Check CUDA visibility
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Re-export if needed
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

### Memory Issues
- Reduce resolution in preprocessing
- Lower batch size (if implemented)
- Ensure no other processes using GPU

### Performance Issues
- Check GPU utilization: `nvidia-smi -l 1`
- Verify XFormers is enabled (check logs)
- Adjust similarity threshold for more frame skipping

## 📁 File Structure

```
enhanced-diffusion/
├── install.sh                    # Installation script
├── requirements.txt             # Python dependencies
├── server_dotsimulate_enhanced.py # Main server
├── run.sh                       # Run script (created by install.sh)
├── test_cuda.py                 # CUDA test (created by install.sh)
└── README.md                    # This file
```

## 🔧 Advanced Configuration

### Environment Variables
```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Torch settings
export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Diffusers cache
export HF_HOME=/workspace/cache
```

### Model Options
The default model is LCM_Dreamshaper_v7. To use a different model, modify the `model_id` in the script:

```python
model_id = "stabilityai/sd-turbo"  # For faster generation
model_id = "runwayml/stable-diffusion-v1-5"  # For standard SD
```

## 📈 Monitoring

Monitor GPU usage and performance:
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Check server logs
tail -f server.log  # If logging is implemented
```

## 🎯 Event Setup Recommendations

### Hardware
- **PC**: High-end with good CPU for video encoding
- **Camera**: HD webcam with good low-light performance
- **Display**: Large vertical monitor (43"+ recommended)
- **Network**: Stable connection, 10+ Mbps upload

### Physical Setup
- **Lighting**: Bright, even frontal lighting
- **Background**: Neutral, non-reflective surface
- **Signage**: Clear positioning instructions
- **Space**: 2-3 meter depth for user movement

## 🤝 Credits

Based on StreamDiffusion research and inspired by DotSimulate's TouchDesigner implementation techniques.

## 📝 License

This implementation is for educational and creative purposes. Please respect the licenses of the underlying models and libraries.