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

# GPU-ACCELERATED VERSION (Recommended for RTX/CUDA systems)
# Ultra-fast GPU processing with PyTorch CUDA
python gpu_upscaler.py "image.jpg" --force-gpu

# GPU batch processing
python gpu_upscaler.py "image_folder" --force-gpu

# Auto-select GPU if available (fallback to CPU OpenCV)
python gpu_upscaler.py "image.jpg" -m ESPCN

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
python -c "import cv2; import PIL; import numpy as np; print('✓ All dependencies imported successfully')"

# Check CUDA availability (for GPU acceleration)
python -c "import cv2; print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')"

# Test OpenCV DNN super-resolution support
python -c "import cv2; print('✓ DNN SuperRes available' if hasattr(cv2.dnn_superres, 'DnnSuperResImpl_create') else '✗ DNN SuperRes NOT available')"

# Verify model downloads
dir models\

# Quick test with specific model (without processing)
python -c "from simple_upscaler import SimpleImageUpscaler; u = SimpleImageUpscaler('ESPCN'); print('✓ Model OK' if u.initialize_model() else '✗ Model FAILED')"

# Test image format support
python -c "from simple_upscaler import SimpleImageUpscaler; u = SimpleImageUpscaler(); print(f'Supported formats: {sorted(u.supported_formats)}')"
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
- Models dictionary maps names to GitHub URLs and scale factors
- Automatic model download and caching in `models/` directory

**Strategy Pattern**: Different AI models implement same upscaling interface
- All models use OpenCV DNN Super-Resolution with 4x scaling
- Device selection abstracted (auto-detect CUDA, fallback to CPU)

**Template Method**: Common processing pipeline with model-specific implementations
- Consistent workflow: initialize → load image → upscale → save with quality control
- Error handling and validation at each step

**Progress Tracking**: Uses tqdm for batch processing feedback
- Real-time progress bars with file names and processing stats
- Graceful error handling continues batch processing on individual failures

### File Structure
```
AI-Image-Upscaler/
├── simple_upscaler.py     # Main application and SimpleImageUpscaler class
├── gpu_upscaler.py        # GPU-accelerated version with PyTorch CUDA support
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
- **ESPCN model**: Recommended for development (~0.5s processing, ~100KB model size)
- **EDSR model**: Use for final high-quality outputs (~4+ minutes per image, ~38MB model)
- **FSRCNN model**: Balanced option (~30 seconds processing time)
- **LapSRN model**: Good for detailed images with moderate processing time
- **GPU acceleration**: Automatically enabled when CUDA is available (significant speedup)
- **Memory management**: Models handle large images through automatic optimization
- **Processing times**: Scale with input resolution (900x1200 → 3600x4800 benchmarks above)

### Debugging and Troubleshooting
```powershell
# Debug model download issues
dir models\  # Check if models downloaded correctly

# Test minimal OpenCV functionality
python -c "import cv2, numpy as np; img=np.zeros((100,100,3), dtype=np.uint8); print(f'OpenCV basic test: {img.shape}')"

# Check specific model initialization without processing
python -c "from simple_upscaler import SimpleImageUpscaler; u = SimpleImageUpscaler('ESPCN'); u.initialize_model()"

# Test with a small test image (create minimal test case)
python -c "import cv2, numpy as np; cv2.imwrite('test_small.jpg', np.random.randint(0,255,(50,50,3),dtype=np.uint8))"
python simple_upscaler.py "test_small.jpg" -m ESPCN

# Check available OpenCV backends
python -c "import cv2; print(f'Available backends: {[cv2.dnn.getAvailableBackends()]}')"
```

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
- **New AI models**: Add to `models` dictionary in `SimpleImageUpscaler.__init__` with URL and scale factor
- **Custom processing**: Override `upscale_image` method for specialized workflows
- **Output formats**: Extend `is_supported_format` and save logic in `upscale_image`
- **Device backends**: Modify device selection logic in `initialize_model` for custom backends
- **Progress callbacks**: Extend tqdm usage in `upscale_batch` for custom progress handling
- **Quality settings**: Customize compression settings in save logic (lines 167-172)
- **Batch processing**: Modify file discovery logic in `upscale_batch` for custom filtering

## Common Issues and Solutions

### Setup Issues
- **"Module 'cv2' has no attribute 'dnn_superres'"**: Install `opencv-contrib-python` instead of basic `opencv-python`
- **PyTorch not detected in setup.py**: The setup script tries to detect CUDA support but falls back gracefully
- **Virtual environment activation**: Use correct script for your shell (PowerShell vs Command Prompt)

### Runtime Issues
- **Model download failures**: Check internet connection and GitHub accessibility
- **CUDA errors**: Automatic fallback to CPU, but check CUDA installation if GPU acceleration needed
- **Memory issues**: Large images may require significant RAM; process smaller batches or use ESPCN model
- **Batch processing stops**: Individual file failures don't stop batch processing; check summary output
