# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

AI Image Upscaler is a Python application that uses deep learning models to enhance images by 4x their original resolution. It leverages OpenCV's DNN super-resolution capabilities with pre-trained models (EDSR, ESPCN, FSRCNN, LapSRN) for high-quality image upscaling.

## Development Environment Setup

### Initial Setup
```powershell
# Setup virtual environment and dependencies
python setup.py
# OR
setup.bat

# Activate virtual environment (PowerShell)
ai_upscaler_env\Scripts\Activate.ps1

# Activate virtual environment (Command Prompt)
ai_upscaler_env\Scripts\activate.bat
```

### Dependencies Installation
```powershell
# Install all dependencies (if setup.py wasn't used)
pip install -r requirements.txt

# Core dependencies
pip install opencv-contrib-python>=4.5.0 numpy>=1.21.0 Pillow>=9.0.0 requests>=2.28.0 tqdm>=4.64.0
```

## Common Development Commands

### Running the Application
```powershell
# Basic single image upscaling (using EDSR model)
python simple_upscaler.py "image.jpg"

# Fast processing with ESPCN model (recommended for development)
python simple_upscaler.py "image.jpg" -m ESPCN

# Batch process a directory
python simple_upscaler.py "image_folder" -m ESPCN

# Custom output path with quality setting
python simple_upscaler.py "input.jpg" -o "output_4k.jpg" -m ESPCN -q 90

# Process with recursive directory scanning
python simple_upscaler.py "image_folder" -m ESPCN -r
```

### Testing and Development
```powershell
# Test model initialization and dependencies
python -c "import torch; import cv2; import PIL; print('✓ All dependencies imported successfully')"

# Check CUDA availability (for GPU acceleration)
python -c "import cv2; print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"

# Verify model downloads
dir models\
```

### Model Management
```powershell
# Models are auto-downloaded to models/ directory:
# - EDSR_x4.pb (38MB) - Highest quality, slowest
# - ESPCN_x4.pb (100KB) - Fastest processing, good quality  
# - FSRCNN_x4.pb - Balanced speed/quality
# - LapSRN_x4.pb - Good for detailed images
```

## Architecture Overview

### Core Components

**SimpleImageUpscaler Class** (`simple_upscaler.py`)
- Main orchestrator for AI upscaling operations
- Handles model initialization, download, and device selection
- Supports multiple AI models (EDSR, ESPCN, FSRCNN, LapSRN)
- Manages both single image and batch processing workflows

**Model Architecture**
- Uses OpenCV DNN Super-Resolution (`cv2.dnn_superres`)
- Downloads pre-trained TensorFlow models (.pb format) from GitHub
- Automatic fallback from GPU to CPU processing
- 4x upscaling factor for all supported models

**Processing Pipeline**
1. Model initialization and download (if needed)
2. Device selection (auto-detect CUDA, fallback to CPU)
3. Image loading and validation (supports JPG, PNG, BMP, TIFF, WebP)
4. AI upscaling using selected model
5. Output saving with quality optimization

### Key Design Patterns

**Model Factory Pattern**: Dynamic model loading based on method parameter
**Strategy Pattern**: Different AI models implement same upscaling interface  
**Template Method**: Common processing pipeline with model-specific implementations
**Progress Tracking**: Uses tqdm for batch processing feedback

### File Structure
```
AI-Image-Upscaler/
├── simple_upscaler.py     # Main application and SimpleImageUpscaler class
├── setup.py               # Environment setup and dependency installation
├── setup.bat              # Windows setup wrapper
├── requirements.txt       # Python dependencies
├── models/                # AI models (auto-downloaded)
│   ├── EDSR_x4.pb        # High quality model (~38MB)
│   └── ESPCN_x4.pb       # Fast processing model (~100KB)
└── ai_upscaler_env/       # Virtual environment (created by setup)
```

## Development Notes

### Performance Considerations
- **ESPCN model**: Recommended for development (sub-second processing)
- **EDSR model**: Use for final high-quality outputs (several minutes per image)
- **GPU acceleration**: Automatically enabled when CUDA is available
- **Memory management**: Models handle large images through automatic optimization

### Supported Input Formats
- Image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
- Processing: Single images or batch directories (with optional recursion)
- Output: Same format as input with quality control for JPEG

### Error Handling
- Automatic model downloading with fallback error messages
- Device compatibility checking (CUDA -> CPU fallback)
- Input validation for file formats and paths
- Progress tracking with detailed error reporting for batch operations

### Extension Points
- New AI models: Add to `models` dictionary in `SimpleImageUpscaler.__init__`
- Custom processing: Override `upscale_image` method for specialized workflows
- Output formats: Extend `is_supported_format` and save logic in `upscale_image`