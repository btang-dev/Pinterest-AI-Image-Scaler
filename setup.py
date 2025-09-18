#!/usr/bin/env python3
"""
Setup script for AI Image Upscaler
This script helps with initial setup and dependency installation
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ {description} failed: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("✗ Python 3.7 or higher is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_virtual_environment():
    """Create a virtual environment."""
    venv_name = "ai_upscaler_env"
    if os.path.exists(venv_name):
        print(f"✓ Virtual environment '{venv_name}' already exists")
        return True
    
    return run_command(f"python -m venv {venv_name}", "Creating virtual environment")

def install_dependencies():
    """Install required dependencies."""
    venv_name = "ai_upscaler_env"
    
    if platform.system() == "Windows":
        pip_path = f"{venv_name}\\Scripts\\pip"
        python_path = f"{venv_name}\\Scripts\\python"
    else:
        pip_path = f"{venv_name}/bin/pip"
        python_path = f"{venv_name}/bin/python"
    
    # Upgrade pip first
    if not run_command(f"{python_path} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install PyTorch with CUDA support if available
    print("🔍 Detecting CUDA availability...")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print("✓ CUDA detected - installing PyTorch with CUDA support")
            torch_command = f"{pip_path} install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        else:
            print("! CUDA not detected - installing CPU-only PyTorch")
            torch_command = f"{pip_path} install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    except ImportError:
        print("! PyTorch not installed - installing CPU version (you can upgrade later)")
        torch_command = f"{pip_path} install torch torchvision"
    
    if not run_command(torch_command, "Installing PyTorch"):
        return False
    
    # Install other dependencies
    return run_command(f"{pip_path} install -r requirements.txt", "Installing other dependencies")

def test_installation():
    """Test if the installation works."""
    venv_name = "ai_upscaler_env"
    
    if platform.system() == "Windows":
        python_path = f"{venv_name}\\Scripts\\python"
    else:
        python_path = f"{venv_name}/bin/python"
    
    test_command = f"{python_path} -c \"import torch; import cv2; import PIL; print('✓ All dependencies imported successfully')\""
    return run_command(test_command, "Testing installation")

def main():
    print("🚀 AI Image Upscaler Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("⚠️  Installation test failed, but dependencies may still work")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    
    if platform.system() == "Windows":
        print("   ai_upscaler_env\\Scripts\\Activate.ps1  (PowerShell)")
        print("   ai_upscaler_env\\Scripts\\activate.bat   (Command Prompt)")
    else:
        print("   source ai_upscaler_env/bin/activate")
    
    print("\n2. Run the upscaler:")
    print("   python upscaler.py --help")
    print("   python upscaler.py \"path/to/your/image.jpg\"")
    
    print("\n3. Read README.md for detailed usage instructions")
    print("\nHappy upscaling! 🎨")

if __name__ == "__main__":
    main()