#!/bin/bash
# Setup script for SmolVLM fine-tuning environment

set -e  # Exit on error

echo "=================================="
echo "SmolVLM Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "[1/7] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if CUDA is available
echo ""
echo "[2/7] Checking CUDA availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo "✓ CUDA is available"
else
    echo "⚠ Warning: nvidia-smi not found. GPU training may not be available."
fi

# Install PyTorch
echo ""
echo "[3/7] Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
echo ""
echo "[4/7] Installing core dependencies..."
pip install transformers datasets accelerate peft trl bitsandbytes

# Install Flash Attention (optional)
echo ""
echo "[5/7] Installing Flash Attention 2 (this may take a few minutes)..."
if pip install flash-attn --no-build-isolation; then
    echo "✓ Flash Attention 2 installed successfully"
else
    echo "⚠ Warning: Flash Attention 2 installation failed. Training will use standard attention (slower)."
    echo "You can continue without it, or try installing manually later."
fi

# Install remaining dependencies
echo ""
echo "[6/7] Installing remaining dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "[7/7] Verifying installation..."
python3 << EOF
import torch
import transformers
import trl
import peft
import bitsandbytes

print('\n✓ PyTorch:', torch.__version__)
print('✓ Transformers:', transformers.__version__)
print('✓ TRL:', trl.__version__)
print('✓ PEFT:', peft.__version__)
print('✓ BitsAndBytes:', bitsandbytes.__version__)
print('✓ CUDA available:', torch.cuda.is_available())

try:
    import flash_attn
    print('✓ Flash Attention:', flash_attn.__version__)
except ImportError:
    print('⚠ Flash Attention: Not installed (optional)')

print('\nAll required packages installed successfully!')
EOF

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x scripts/*.py
chmod +x examples/*.py

echo ""
echo "=================================="
echo "Setup Complete! 🎉"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Configure your paths in config/training/smolvlm_config.yaml"
echo "2. Generate pseudolabels: python scripts/generate_pseudolabels.py"
echo "3. Start training: python scripts/train_smolvlm.py"
echo ""
echo "For more information, see:"
echo "- docs/SMOLVLM_SETUP.md"
echo "- docs/SMOLVLM_FINETUNING.md"
echo ""

