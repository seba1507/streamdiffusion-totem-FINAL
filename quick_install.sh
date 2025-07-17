#!/bin/bash
# Quick install script for RunPod - Just copy and paste!

cd /workspace && \
rm -rf streamdiffusion-totem-FINAL && \
echo "📥 Cloning repository..." && \
git clone https://github.com/seba1507/streamdiffusion-totem-FINAL.git && \
cd streamdiffusion-totem-FINAL && \
chmod +x install.sh run.sh && \
echo "🔧 Installing dependencies..." && \
./install.sh && \
echo "" && \
echo "===================================================================" && \
echo "✅ Installation complete!" && \
echo "===================================================================" && \
echo "" && \
echo "🚀 Starting server..." && \
./run.sh
