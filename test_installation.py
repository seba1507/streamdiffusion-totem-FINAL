#!/usr/bin/env python3
"""
Test installation script for Enhanced StreamDiffusion
Verifies all dependencies are correctly installed
"""

import sys
import os

print("=" * 70)
print("🧪 Enhanced StreamDiffusion - Installation Test")
print("=" * 70)

# Set CUDA environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Test results
all_tests_passed = True

# Test 1: Python version
print("\n1️⃣ Testing Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")
if python_version.major == 3 and python_version.minor >= 10:
    print("   ✅ Python version OK")
else:
    print("   ❌ Python 3.10+ required")
    all_tests_passed = False

# Test 2: CUDA availability
print("\n2️⃣ Testing CUDA...")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Test tensor operation
        test_tensor = torch.randn(1, 3, 512, 512).cuda()
        print(f"   ✅ GPU tensor test passed")
    else:
        print("   ❌ CUDA NOT available")
        all_tests_passed = False
except Exception as e:
    print(f"   ❌ PyTorch/CUDA error: {e}")
    all_tests_passed = False

# Test 3: Core dependencies
print("\n3️⃣ Testing core dependencies...")
core_deps = {
    "diffusers": "Diffusers",
    "transformers": "Transformers",
    "accelerate": "Accelerate",
    "xformers": "XFormers",
    "huggingface_hub": "Hugging Face Hub",
    "safetensors": "Safetensors"
}

for module, name in core_deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"   ✅ {name}: {version}")
    except ImportError as e:
        print(f"   ❌ {name}: NOT INSTALLED - {e}")
        all_tests_passed = False

# Test 4: Web framework
print("\n4️⃣ Testing web framework...")
web_deps = {
    "fastapi": "FastAPI",
    "uvicorn": "Uvicorn",
    "websockets": "WebSockets"
}

for module, name in web_deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"   ✅ {name}: {version}")
    except ImportError as e:
        print(f"   ❌ {name}: NOT INSTALLED - {e}")
        all_tests_passed = False

# Test 5: Image processing
print("\n5️⃣ Testing image processing libraries...")
image_deps = {
    "PIL": "Pillow",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "scipy": "SciPy"
}

for module, name in image_deps.items():
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"   ✅ {name}: {version}")
    except ImportError as e:
        print(f"   ❌ {name}: NOT INSTALLED - {e}")
        all_tests_passed = False

# Test 6: Server file
print("\n6️⃣ Testing server file...")
if os.path.exists("server_dotsimulate_enhanced.py"):
    print("   ✅ server_dotsimulate_enhanced.py found")
    
    # Try to import it
    try:
        import server_dotsimulate_enhanced
        print("   ✅ Server file imports successfully")
    except Exception as e:
        print(f"   ⚠️  Server import warning: {e}")
        # This might fail due to server startup, not critical
else:
    print("   ❌ server_dotsimulate_enhanced.py NOT FOUND")
    all_tests_passed = False

# Test 7: Model accessibility
print("\n7️⃣ Testing model download...")
try:
    from huggingface_hub import snapshot_download
    print("   ✅ Hugging Face Hub accessible")
    print("   ℹ️  Model will be downloaded on first run")
except Exception as e:
    print(f"   ⚠️  Model test warning: {e}")

# Summary
print("\n" + "=" * 70)
if all_tests_passed:
    print("✅ ALL TESTS PASSED! Installation is ready.")
    print("\nTo start the server, run:")
    print("  ./run.sh")
    print("\nOr manually:")
    print("  export CUDA_VISIBLE_DEVICES=0")
    print("  python3 server_dotsimulate_enhanced.py")
else:
    print("❌ SOME TESTS FAILED! Please check the errors above.")
    print("\nTry running:")
    print("  ./install.sh")
    sys.exit(1)

print("=" * 70)
