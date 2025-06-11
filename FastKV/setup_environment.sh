#!/bin/bash

set -e  # Exit on any error

echo "Starting EchoCache environment setup..."
echo "================================================"

# Step 1: Create conda environment
echo "Creating conda environment 'echocache' with Python 3.9..."
conda create -n echocache python=3.9 -y

# Step 2: Activate environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate echocache

# Step 3: Install requirements
echo "Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found in current directory"
fi

# Step 4: Install flash-attn
echo "Installing flash-attn (this may take a while)..."
pip install flash-attn --no-build-isolation

# Step 5: Build baseline/adakv
echo "Building baseline/adakv..."
if [ -d "baseline/adakv" ]; then
    cd baseline/adakv
    make
    cd ../..
    echo "COMPLETED: baseline/adakv build"
else
    echo "WARNING: baseline/adakv directory not found"
fi

# Step 6: Install fast-hadamard-transform
echo "Installing fast-hadamard-transform..."
if [ -d "fast-hadamard-transform" ]; then
    cd fast-hadamard-transform/
    python setup.py install
    cd ..
    echo "COMPLETED: fast-hadamard-transform installation"
else
    echo "WARNING: fast-hadamard-transform directory not found"
fi

# Step 7: Install qtip kernels
echo "Installing qtip kernels..."
if [ -d "qtip/qtip-kernels" ]; then
    cd qtip/qtip-kernels/
    python setup.py install
    cd ../..
    echo "COMPLETED: qtip kernels installation"
else
    echo "WARNING: qtip/qtip-kernels directory not found"
fi

echo "================================================"
echo "EchoCache environment setup complete."
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate echocache"
echo ""
echo "Current Python location: $(which python)"
echo "Current environment: $CONDA_DEFAULT_ENV"
