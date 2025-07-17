#!/bin/bash

# RunPod Quick Deploy Script for Enhanced StreamDiffusion
# Usage: Just paste this entire script in RunPod terminal

echo "===================================================================="
echo "🚀 RunPod Quick Deploy - Enhanced StreamDiffusion"
echo "===================================================================="

# Function to clone repository
clone_repo() {
    local REPO_URL=$1
    local IS_PRIVATE=$2
    
    if [ "$IS_PRIVATE" = "true" ]; then
        echo "🔐 Private repository detected"
        echo "Please enter your GitHub credentials:"
        echo "Username: "
        read GITHUB_USER
        echo "Personal Access Token (will be hidden): "
        read -s GITHUB_TOKEN
        
        git clone https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/streamdiffusion-totem-FINAL.git
    else
        git clone $REPO_URL
    fi
}

# Navigate to workspace
cd /workspace

# Check if repo already exists
if [ -d "streamdiffusion-totem-FINAL" ]; then
    echo "📁 Repository already exists. Updating..."
    cd streamdiffusion-totem-FINAL
    git pull
else
    echo "📥 Cloning repository..."
    # Try public clone first
    if ! git clone https://github.com/seba1507/streamdiffusion-totem-FINAL.git 2>/dev/null; then
        echo "Repository appears to be private."
        clone_repo "https://github.com/seba1507/streamdiffusion-totem-FINAL.git" "true"
    fi
    cd streamdiffusion-totem-FINAL
fi

# Make scripts executable
chmod +x install.sh run.sh deploy_runpod.sh 2>/dev/null || true

# Run installation
echo ""
echo "🔧 Starting installation..."
./install.sh

# If installation successful, offer to start server
if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================================="
    echo "✅ Deployment complete!"
    echo "===================================================================="
    echo ""
    echo "Start the server now? (y/n)"
    read -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./run.sh
    else
        echo "To start the server later, run: ./run.sh"
    fi
else
    echo "❌ Installation failed. Please check errors above."
fi
