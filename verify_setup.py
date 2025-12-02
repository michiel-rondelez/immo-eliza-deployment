#!/usr/bin/env python3
"""
Verification script to check if all dependencies are properly installed.
Run this after setup.sh to verify your installation.
"""

import sys
from pathlib import Path

def check_package(package_name, import_name=None, show_version=True):
    """Check if a package is installed and optionally show its version."""
    if import_name is None:
        import_name = package_name

    try:
        module = __import__(import_name)
        if show_version and hasattr(module, '__version__'):
            print(f"✓ {package_name:20s} {module.__version__}")
        else:
            print(f"✓ {package_name:20s} installed")
        return True
    except ImportError:
        print(f"✗ {package_name:20s} NOT INSTALLED")
        return False

def main():
    """Run verification checks."""
    print("=" * 60)
    print("Immo Eliza ML Pipeline - Setup Verification")
    print("=" * 60)
    print()

    # Check Python version
    python_version = sys.version.split()[0]
    print(f"Python version: {python_version}")
    print()

    # Check core packages
    print("Checking dependencies:")
    print("-" * 60)

    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'streamlit': 'streamlit',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'pydantic': 'pydantic',
        'xgboost': 'xgboost',
    }

    results = {}
    for package_name, import_name in packages.items():
        results[package_name] = check_package(package_name, import_name)

    print()
    print("-" * 60)

    # Check project structure
    print()
    print("Checking project structure:")
    print("-" * 60)

    required_files = [
        'src/preprocessing.py',
        'src/models.py',
        'src/main.py',
        'streamlit_app.py',
        'requirements.txt',
        'setup.sh',
        'run_api.sh',
        'run_streamlit.sh',
    ]

    all_files_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} NOT FOUND")
            all_files_exist = False

    print()
    print("=" * 60)

    # Summary
    failed_packages = [pkg for pkg, success in results.items() if not success]

    if not failed_packages and all_files_exist:
        print("✓ All checks passed! Your setup is complete.")
        print()
        print("Next steps:")
        print("  - Run Streamlit: ./run_streamlit.sh")
        print("  - Run FastAPI:   ./run_api.sh")
    else:
        print("⚠️  Some issues were found:")
        if failed_packages:
            print(f"  Missing packages: {', '.join(failed_packages)}")
            if 'xgboost' in failed_packages:
                print()
                print("  XGBoost installation failed. The app will work without it.")
                print("  To install XGBoost:")
                print("    - macOS:   brew install cmake && pip install xgboost")
                print("    - Linux:   sudo apt-get install cmake build-essential && pip install xgboost")
                print("    - Windows: pip install xgboost")
        if not all_files_exist:
            print("  Some project files are missing.")

    print("=" * 60)

    # Test imports from project modules
    print()
    print("Testing project module imports:")
    print("-" * 60)

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from preprocessing import FeaturePreprocessor
        print("✓ preprocessing.FeaturePreprocessor")
    except Exception as e:
        print(f"✗ preprocessing.FeaturePreprocessor - {e}")

    try:
        from models import ModelTrainer, XGBOOST_AVAILABLE
        print(f"✓ models.ModelTrainer (XGBoost available: {XGBOOST_AVAILABLE})")
    except Exception as e:
        print(f"✗ models.ModelTrainer - {e}")

    try:
        from main import app
        print("✓ main.app (FastAPI)")
    except Exception as e:
        print(f"✗ main.app (FastAPI) - {e}")

    print("=" * 60)
    print()

if __name__ == "__main__":
    main()
