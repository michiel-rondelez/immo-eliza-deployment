# Setup Guide

This guide will help you set up the Immo Eliza ML Pipeline project from scratch, with special attention to XGBoost installation.

## Quick Start (Automated)

The easiest way to set up the project is to use the automated setup script:

```bash
# Run the setup script
./setup.sh

# Verify installation
python3 verify_setup.py
```

## Manual Setup

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

**For XGBoost (optional but recommended):**
- **macOS**: `brew install cmake`
- **Linux (Ubuntu/Debian)**: `sudo apt-get install build-essential cmake`
- **Windows**: Visual Studio Build Tools

### 2. Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Run verification script
python3 verify_setup.py
```

## XGBoost Installation Issues

If you encounter errors installing XGBoost, try these solutions:

### Common Error: "ModuleNotFoundError: No module named 'xgboost'"

**Solution 1: Install with pip**
```bash
pip install xgboost
```

**Solution 2: Install with conda (if using Anaconda)**
```bash
conda install -c conda-forge xgboost
```

**Solution 3: Install from source (macOS/Linux)**
```bash
# Install cmake first
# macOS:
brew install cmake

# Linux:
sudo apt-get install cmake build-essential

# Then install xgboost
pip install xgboost
```

**Solution 4: Use pre-built wheels**
```bash
# For specific Python version, e.g., Python 3.9
pip install xgboost==2.0.3
```

### macOS Specific Issues

If you see "library not loaded" errors:

```bash
# Install libomp
brew install libomp

# Reinstall xgboost
pip uninstall xgboost
pip install xgboost
```

### Windows Specific Issues

If you see compiler errors:

1. Install Visual Studio Build Tools from: https://visualstudio.microsoft.com/downloads/
2. During installation, select "Desktop development with C++"
3. Restart your terminal and try again:
   ```bash
   pip install xgboost
   ```

### Linux Specific Issues

If you see "gcc" or "g++" not found:

```bash
# Install build tools
sudo apt-get update
sudo apt-get install build-essential cmake

# Install xgboost
pip install xgboost
```

## Running Without XGBoost

**Good news:** The application is designed to work even if XGBoost is not installed!

If XGBoost installation fails:
1. The app will automatically detect this
2. It will show a warning message
3. You can still use other models:
   - Linear Regression
   - Support Vector Regression (SVR)
   - Decision Tree
   - Random Forest

To use the app without XGBoost:

```bash
# Just skip XGBoost installation and continue
pip install numpy pandas scikit-learn fastapi uvicorn streamlit matplotlib seaborn pydantic python-multipart

# Run the app
./run_streamlit.sh
```

## Verifying Your Installation

Run the verification script to check everything:

```bash
python3 verify_setup.py
```

You should see:
```
✓ numpy               1.26.x
✓ pandas              2.1.x
✓ scikit-learn        1.4.x
✓ xgboost             2.0.x
✓ fastapi             0.109.x
✓ streamlit           1.30.x
...
✓ All checks passed!
```

If XGBoost shows "NOT INSTALLED", don't worry - the app will still work with other models.

## Running the Application

### Option 1: Streamlit UI (Recommended for beginners)

```bash
./run_streamlit.sh
```

Open your browser at `http://localhost:8501`

### Option 2: FastAPI Backend

```bash
./run_api.sh
```

API docs at `http://localhost:8000/docs`

### Option 3: Both

```bash
# Terminal 1
./run_api.sh

# Terminal 2
./run_streamlit.sh
```

## Troubleshooting

### Issue: "Permission denied" when running scripts

**Solution:**
```bash
chmod +x setup.sh run_api.sh run_streamlit.sh verify_setup.py
```

### Issue: "Port already in use"

**Solution:**
```bash
# Find and kill the process using the port
# For Streamlit (port 8501):
lsof -ti:8501 | xargs kill

# For FastAPI (port 8000):
lsof -ti:8000 | xargs kill
```

### Issue: "No module named 'src'"

**Solution:**
Make sure you're running commands from the project root directory.

```bash
cd /path/to/immo-eliza-deployment
./run_streamlit.sh
```

### Issue: Packages installed but verification fails

**Solution:**
Make sure you're using the correct Python interpreter:

```bash
# Check which Python you're using
which python3

# Check if it's from your virtual environment
# Should show something like: /path/to/venv/bin/python3
```

## Getting Help

If you continue to have issues:

1. Check the error message carefully
2. Make sure you're in a virtual environment
3. Try upgrading pip: `pip install --upgrade pip`
4. Try reinstalling: `pip uninstall <package> && pip install <package>`
5. Check the GitHub issues for similar problems

## Next Steps

Once setup is complete:

1. ✓ Read the main [README.md](README.md) for usage instructions
2. ✓ Prepare your data (see Data Format section in README)
3. ✓ Run the Streamlit app and try training models
4. ✓ Explore the API documentation at `/docs`

---

**Note:** This project is designed to be resilient. Even if some optional dependencies fail to install, core functionality will still work!
