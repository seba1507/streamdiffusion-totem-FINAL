#!/bin/bash

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Get hostname for RunPod URL
HOSTNAME=$(hostname)
RUNPOD_URL="https://${HOSTNAME}-8000.proxy.runpod.net/"

# Clear screen for clean output
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

# Check if server file exists
if [ ! -f "server_dotsimulate_enhanced.py" ]; then
    echo "❌ ERROR: server_dotsimulate_enhanced.py not found!"
    echo "Please make sure you're in the correct directory."
    exit 1
fi

# Start the server
python3 server_dotsimulate_enhanced.py
