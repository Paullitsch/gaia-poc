#!/bin/bash
# GAIA Worker Setup Script
# Run this on your local PC with GPU

set -e

echo "╔══════════════════════════════════════════╗"
echo "║      GAIA Worker Setup                   ║"
echo "╚══════════════════════════════════════════╝"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+"
    exit 1
fi
echo "✅ Python: $(python3 --version)"

# Check NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  nvidia-smi not found. GPU may not be available."
    echo "   Install NVIDIA drivers for GPU acceleration."
fi

# Create venv
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA
echo ""
echo "📦 Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install other deps
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Verify
echo ""
echo "🔍 Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
import gymnasium
print(f'Gymnasium: {gymnasium.__version__}')
import numpy
print(f'NumPy: {numpy.__version__}')
print('✅ All good!')
"

# Create config if not exists
if [ ! -f config.yaml ]; then
    cp config.example.yaml config.yaml
    echo ""
    echo "📝 Created config.yaml from template"
    echo "   Edit it to set your VPS connection details"
fi

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║  Setup complete! Run experiments:        ║"
echo "║                                          ║"
echo "║  source venv/bin/activate                ║"
echo "║  python run_all.py                       ║"
echo "║  python run_all.py --quick   (test run)  ║"
echo "║  python run_all.py --method cma_es       ║"
echo "╚══════════════════════════════════════════╝"
