#!/bin/bash

echo "===================================================================="
echo "🚀 INSTALACIÓN COMPLETA PARA POD NUEVO - STREAMDIFFUSION"
echo "===================================================================="

# Configurar variables de entorno
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH

# Verificar CUDA
echo "🔍 Verificando CUDA..."
nvidia-smi

# Actualizar pip
echo "📦 Actualizando pip..."
python3 -m pip install --upgrade pip

# Instalar dependencias del sistema si es necesario
echo "📦 Instalando dependencias del sistema..."
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget

# Instalar desde requirements.txt
echo "📦 Instalando paquetes Python..."
pip install -r requirements.txt

# Verificar instalación
echo "🔍 Verificando instalación..."
python3 -c "
import torch
import tensorrt
import polygraphy
import diffusers
import streamdiffusion

print('✅ PyTorch:', torch.__version__)
print('✅ TensorRT:', tensorrt.__version__)
print('✅ CUDA disponible:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✅ GPU:', torch.cuda.get_device_name(0))
print('✅ Todas las dependencias instaladas correctamente!')
"

echo "===================================================================="
echo "✅ Instalación completa!"
echo "🚀 Ejecuta: python3 server_optimizado_v11.py"
echo "===================================================================="
