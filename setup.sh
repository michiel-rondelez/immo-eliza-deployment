#!/bin/bash

# Setup script for Immo Eliza ML Pipeline
# This script ensures all dependencies are properly installed

set -e  # Exit on error

echo "=================================================="
echo "Immo Eliza ML Pipeline - Setup Script"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Running in virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  WARNING: Not running in a virtual environment"
    echo "   It's recommended to use a virtual environment"
    echo ""
    read -p "Do you want to create a virtual environment? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        echo "Activating virtual environment..."
        source venv/bin/activate
        echo "✓ Virtual environment created and activated"
    fi
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "=================================================="
echo "Verifying installation..."
echo "=================================================="
echo ""

# Check each package
packages=("fastapi" "uvicorn" "pandas" "numpy" "scikit-learn" "streamlit" "matplotlib" "seaborn" "pydantic")

all_ok=true
for package in "${packages[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package installed successfully"
    else
        echo "✗ $package installation FAILED"
        all_ok=false
    fi
done

# Special check for xgboost
echo ""
echo "Checking XGBoost installation..."
if python3 -c "import xgboost" 2>/dev/null; then
    version=$(python3 -c "import xgboost; print(xgboost.__version__)")
    echo "✓ xgboost $version installed successfully"
else
    echo "✗ xgboost installation FAILED"
    echo ""
    echo "Trying alternative installation methods..."

    # Try installing with pip directly
    echo "Attempting: pip install xgboost"
    if pip install xgboost; then
        echo "✓ xgboost installed successfully"
    else
        echo "✗ Failed to install xgboost"
        echo ""
        echo "XGBoost installation failed. This could be due to:"
        echo "  1. Missing system dependencies (cmake, gcc, etc.)"
        echo "  2. Incompatible Python version"
        echo "  3. Platform-specific issues"
        echo ""
        echo "The application will still work with other models."
        echo "To install XGBoost manually:"
        echo "  - macOS: brew install cmake && pip install xgboost"
        echo "  - Linux: sudo apt-get install build-essential cmake && pip install xgboost"
        echo "  - Windows: pip install xgboost (ensure Visual Studio Build Tools installed)"
        all_ok=false
    fi
fi

echo ""
echo "=================================================="
if [ "$all_ok" = true ]; then
    echo "✓ Setup completed successfully!"
else
    echo "⚠️  Setup completed with warnings"
    echo "   Some packages may not be installed correctly"
fi
echo "=================================================="
echo ""
echo "Next steps:"
echo "  1. To run Streamlit UI:  ./run_streamlit.sh"
echo "  2. To run FastAPI:       ./run_api.sh"
echo "  3. To verify setup:      python3 verify_setup.py"
echo ""
