#!/usr/bin/env python3
"""
GPU-Accelerated AI Image Upscaler - 4x Image Super Resolution
Enhanced version with PyTorch GPU support for faster processing
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
import time
import urllib.request

class ESRGANModel(nn.Module):
    """Simple ESRGAN-style model for 4x upscaling using PyTorch"""
    def __init__(self):
        super(ESRGANModel, self).__init__()
        # Simple upscaling network - in production you'd use a pre-trained model
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bicubic', align_corners=False)
        self.final_conv = nn.Conv2d(64, 3, 9, padding=4)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.final_conv(x))
        return x

class GPUImageUpscaler:
    def __init__(self, method='ESPCN', device='auto', use_pytorch_gpu=True):
        """
        Initialize the GPU-accelerated image upscaler.
        
        Args:
            method (str): Super-resolution method ('ESPCN', 'EDSR', 'FSRCNN', 'LapSRN', 'PYTORCH')
            device (str): Device preference ('auto', 'cuda', 'cpu')
            use_pytorch_gpu (bool): Use PyTorch GPU acceleration when available
        """
        self.method = method
        self.device = device
        self.use_pytorch_gpu = use_pytorch_gpu
        self.sr = None
        self.pytorch_model = None
        self.model_loaded = False
        self.gpu_available = torch.cuda.is_available()
        self.torch_device = torch.device('cuda' if self.gpu_available and device != 'cpu' else 'cpu')
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Model URLs for OpenCV DNN super-resolution models
        self.models = {
            'EDSR': {
                'url': 'https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb',
                'scale': 4
            },
            'ESPCN': {
                'url': 'https://github.com/fannymonori/TF-ESPCN/raw/master/export/ESPCN_x4.pb', 
                'scale': 4
            },
            'FSRCNN': {
                'url': 'https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb',
                'scale': 4  
            },
            'LapSRN': {
                'url': 'https://github.com/fannymonori/TF-LapSRN/raw/master/export/LapSRN_x4.pb',
                'scale': 4
            }
        }
        
        print(f"🎮 GPU Status: {'✅ Available' if self.gpu_available else '❌ Not Available'}")
        if self.gpu_available:
            print(f"🔥 GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def download_model(self, model_name):
        """Download the AI model if not already present."""
        if model_name == 'PYTORCH':
            return 'pytorch'  # PyTorch models are initialized in memory
            
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / f"{model_name}_x4.pb"
        
        if model_path.exists():
            print(f"✓ Model {model_name} already exists")
            return str(model_path)
        
        print(f"📥 Downloading {model_name} model...")
        try:
            url = self.models[model_name]['url']
            urllib.request.urlretrieve(url, model_path)
            print(f"✓ Downloaded {model_name} model successfully")
            return str(model_path)
        except Exception as e:
            print(f"✗ Failed to download {model_name} model: {e}")
            return None
    
    def initialize_model(self):
        """Initialize the super-resolution model."""
        print(f"🔧 Initializing {self.method} model...")
        
        # If using PyTorch method or GPU is available and requested
        if self.method == 'PYTORCH' or (self.use_pytorch_gpu and self.gpu_available and self.method in ['ESPCN']):
            return self.initialize_pytorch_model()
        else:
            return self.initialize_opencv_model()
    
    def initialize_pytorch_model(self):
        """Initialize PyTorch-based upscaling model."""
        try:
            print(f"🚀 Initializing PyTorch GPU model on {self.torch_device}")
            
            # For this demo, we'll use PyTorch's built-in bicubic upsampling with some enhancement
            # In production, you'd load a pre-trained ESRGAN or Real-ESRGAN model
            self.pytorch_model = ESRGANModel().to(self.torch_device)
            
            # Initialize with random weights (in production, load pre-trained weights)
            # For now, we'll use a simpler approach with PyTorch's interpolation + sharpening
            self.model_loaded = True
            print(f"✓ PyTorch GPU model initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize PyTorch model: {str(e)}")
            return False
    
    def initialize_opencv_model(self):
        """Initialize OpenCV DNN super-resolution model."""
        try:
            model_path = self.download_model(self.method)
            if not model_path:
                return False
            
            # Create DNN super-resolution object
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            
            # Read and set the model
            self.sr.readModel(model_path)
            self.sr.setModel(self.method.lower(), 4)  # 4x upscaling
            
            # For OpenCV, we'll stick with CPU since CUDA version isn't available
            print("💻 Using CPU processing (OpenCV)")
            self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_loaded = True
            print(f"✓ {self.method} model initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize model: {str(e)}")
            return False
    
    def is_supported_format(self, file_path):
        """Check if the file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def upscale_with_pytorch(self, img_array):
        """Upscale image using PyTorch with GPU acceleration."""
        try:
            # Convert BGR to RGB and normalize
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # BHWC -> BCHW
            img_tensor = img_tensor.to(self.torch_device)
            
            with torch.no_grad():
                # Enhanced bicubic upsampling with sharpening
                upscaled = F.interpolate(img_tensor, scale_factor=4, mode='bicubic', align_corners=False)
                
                # Apply subtle sharpening filter
                sharpen_kernel = torch.tensor([[[[-1, -1, -1],
                                               [-1,  9, -1],
                                               [-1, -1, -1]]]], dtype=torch.float32).to(self.torch_device)
                
                # Apply to each channel
                channels = []
                for i in range(3):
                    channel = upscaled[:, i:i+1, :, :]
                    sharpened = F.conv2d(channel, sharpen_kernel, padding=1)
                    channels.append(sharpened)
                
                upscaled = torch.cat(channels, dim=1)
                upscaled = torch.clamp(upscaled, 0, 1)
                
                # Convert back to numpy
                result = upscaled.squeeze(0).permute(1, 2, 0).cpu().numpy()
                result = (result * 255).astype(np.uint8)
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                
            return result
            
        except Exception as e:
            print(f"  ⚠️ GPU upscaling failed: {e}")
            print("  🔄 Falling back to CPU bicubic interpolation...")
            height, width = img_array.shape[:2]
            return cv2.resize(img_array, (width * 4, height * 4), interpolation=cv2.INTER_CUBIC)
    
    def upscale_image(self, input_path, output_path=None, quality=95):
        """
        Upscale a single image by 4x.
        """
        if not self.model_loaded:
            raise RuntimeError("Model not initialized. Call initialize_model() first.")
        
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")
        
        if not self.is_supported_format(input_path):
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        
        # Generate output path if not provided
        if output_path is None:
            suffix = "_gpu_upscaled_4x" if self.pytorch_model else "_upscaled_4x"
            output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
        else:
            output_path = Path(output_path)
        
        print(f"📸 Processing: {input_path.name}")
        start_time = time.time()
        
        try:
            # Read image
            img = cv2.imread(str(input_path))
            if img is None:
                raise ValueError(f"Cannot read image: {input_path}")
            
            # Get original dimensions
            original_height, original_width = img.shape[:2]
            print(f"  📏 Original size: {original_width}x{original_height}")
            
            # Upscale image
            if self.pytorch_model and self.gpu_available:
                print(f"  🚀 GPU upscaling with PyTorch...")
                upscaled = self.upscale_with_pytorch(img)
            else:
                print(f"  🔄 Upscaling with {self.method}...")
                upscaled = self.sr.upsample(img)
            
            # Get new dimensions
            new_height, new_width = upscaled.shape[:2]
            print(f"  📏 Upscaled size: {new_width}x{new_height}")
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate quality settings
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif output_path.suffix.lower() == '.png':
                cv2.imwrite(str(output_path), upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 6])
            else:
                cv2.imwrite(str(output_path), upscaled)
            
            processing_time = time.time() - start_time
            device_info = f"GPU ({self.torch_device})" if self.pytorch_model and self.gpu_available else "CPU"
            print(f"  ✅ Completed in {processing_time:.2f}s using {device_info}")
            print(f"  💾 Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"  ❌ Failed to process {input_path.name}: {str(e)}")
            raise
    
    def upscale_batch(self, input_dir, output_dir=None, recursive=False, quality=95):
        """Upscale multiple images in a directory."""
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Set up output directory
        if output_dir is None:
            suffix = "_gpu_upscaled_4x" if self.pytorch_model else "_upscaled_4x"
            output_dir = input_dir / f"upscaled_4x"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all supported image files
        pattern = "**/*" if recursive else "*"
        image_files = []
        
        for pattern_ext in self.supported_formats:
            image_files.extend(input_dir.glob(f"{pattern}{pattern_ext}"))
            image_files.extend(input_dir.glob(f"{pattern}{pattern_ext.upper()}"))
        
        if not image_files:
            print("❌ No supported image files found.")
            return []
        
        device_info = f"GPU ({self.torch_device})" if self.pytorch_model and self.gpu_available else "CPU"
        print(f"📁 Found {len(image_files)} image(s) to process using {device_info}")
        
        # Process images with progress bar
        successful_outputs = []
        failed_files = []
        
        with tqdm(image_files, desc="🚀 GPU Upscaling" if self.gpu_available else "🔄 Upscaling", unit="img") as pbar:
            for img_file in pbar:
                try:
                    # Update progress bar description
                    pbar.set_postfix_str(f"Processing {img_file.name}")
                    
                    # Maintain directory structure
                    relative_path = img_file.relative_to(input_dir)
                    suffix = "_gpu_upscaled_4x" if self.pytorch_model else "_upscaled_4x"
                    output_file = output_dir / relative_path.parent / f"{relative_path.stem}{suffix}{relative_path.suffix}"
                    
                    result = self.upscale_image(img_file, output_file, quality)
                    successful_outputs.append(result)
                    
                except Exception as e:
                    failed_files.append((img_file, str(e)))
                    continue
        
        # Print summary
        print(f"\n📊 Processing Summary:")
        print(f"  ✅ Successfully processed: {len(successful_outputs)}")
        print(f"  ❌ Failed: {len(failed_files)}")
        
        if failed_files:
            print("\n❌ Failed files:")
            for file_path, error in failed_files[:5]:  # Show first 5 failures
                print(f"  - {file_path.name}: {error}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files) - 5} more")
        
        return successful_outputs


def main():
    parser = argparse.ArgumentParser(description="GPU-Accelerated AI Image Upscaler - 4x Super Resolution")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("-m", "--method", default="ESPCN", 
                       choices=["EDSR", "ESPCN", "FSRCNN", "LapSRN", "PYTORCH"],
                       help="AI method to use for upscaling (default: ESPCN, PYTORCH for GPU)")
    parser.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for processing (default: auto)")
    parser.add_argument("-q", "--quality", type=int, default=95, metavar="1-100",
                       help="JPEG quality (1-100, default: 95)")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="Process subdirectories recursively (for directory input)")
    parser.add_argument("--batch", action="store_true",
                       help="Force batch processing mode")
    parser.add_argument("--force-gpu", action="store_true",
                       help="Force GPU acceleration using PyTorch (overrides method)")
    
    args = parser.parse_args()
    
    # Validate quality
    if not 1 <= args.quality <= 100:
        parser.error("Quality must be between 1 and 100")
    
    # Auto-select GPU method if GPU is available and not explicitly set
    if args.force_gpu or (torch.cuda.is_available() and args.device != 'cpu' and args.method == 'ESPCN'):
        args.method = 'PYTORCH'
        print("🎮 Auto-selected PyTorch GPU acceleration")
    
    # Initialize upscaler
    print("🚀 GPU-Accelerated AI Image Upscaler - 4x Super Resolution")
    print("=" * 65)
    print(f"🔧 Method: {args.method}")
    print(f"💻 Device: {args.device}")
    print(f"🎮 PyTorch CUDA: {'✅ Available' if torch.cuda.is_available() else '❌ Not Available'}")
    print()
    
    upscaler = GPUImageUpscaler(method=args.method, device=args.device)
    
    if not upscaler.initialize_model():
        print("\n❌ Failed to initialize model.")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    try:
        if input_path.is_file() and not args.batch:
            # Single image processing
            print(f"\n📸 Processing single image...")
            result = upscaler.upscale_image(args.input, args.output, args.quality)
            print(f"\n🎉 Successfully upscaled image!")
            print(f"📂 Output: {result}")
            
        elif input_path.is_dir() or args.batch:
            # Batch processing
            print(f"\n📁 Processing directory: {input_path}")
            results = upscaler.upscale_batch(
                args.input, 
                args.output, 
                args.recursive, 
                args.quality
            )
            print(f"\n🎉 Batch processing completed!")
            if results:
                print(f"📂 Output directory: {Path(results[0]).parent}")
            
        else:
            print(f"❌ Input path does not exist: {args.input}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()