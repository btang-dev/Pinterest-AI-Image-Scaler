#!/usr/bin/env python3
"""
Simple AI Image Upscaler - 4x Image Super Resolution
A working implementation using advanced OpenCV super-resolution methods
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import requests
from tqdm import tqdm
import time
import urllib.request

class SimpleImageUpscaler:
    def __init__(self, method='EDSR', device='auto'):
        """
        Initialize the image upscaler with specified method and device.
        
        Args:
            method (str): Super-resolution method ('EDSR', 'ESPCN', 'FSRCNN', 'LapSRN')
            device (str): Device preference ('auto', 'cuda', 'cpu')
        """
        self.method = method
        self.device = device
        self.sr = None
        self.model_loaded = False
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
    
    def download_model(self, model_name):
        """Download the AI model if not already present."""
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
        
        try:
            model_path = self.download_model(self.method)
            if not model_path:
                return False
            
            # Create DNN super-resolution object
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            
            # Read and set the model
            self.sr.readModel(model_path)
            self.sr.setModel(self.method.lower(), 4)  # 4x upscaling
            
            # Set preferred backend and target
            if self.device == 'cuda' or (self.device == 'auto' and cv2.cuda.getCudaEnabledDeviceCount() > 0):
                print("🚀 Using GPU acceleration")
                self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                print("💻 Using CPU processing")
                self.sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            self.model_loaded = True
            print(f"✓ {self.method} model initialized successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize model: {str(e)}")
            print("💡 Tip: Make sure you have opencv-contrib-python installed")
            return False
    
    def is_supported_format(self, file_path):
        """Check if the file format is supported."""
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def upscale_image(self, input_path, output_path=None, quality=95):
        """
        Upscale a single image by 4x.
        
        Args:
            input_path (str): Path to input image
            output_path (str): Path to save upscaled image (optional) 
            quality (int): JPEG quality for output (1-100)
        
        Returns:
            str: Path to the upscaled image
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
            output_path = input_path.parent / f"{input_path.stem}_upscaled_4x{input_path.suffix}"
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
            
            # Upscale image using AI model
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
            print(f"  ✅ Completed in {processing_time:.2f}s")
            print(f"  💾 Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"  ❌ Failed to process {input_path.name}: {str(e)}")
            raise
    
    def upscale_batch(self, input_dir, output_dir=None, recursive=False, quality=95):
        """
        Upscale multiple images in a directory.
        
        Args:
            input_dir (str): Directory containing input images
            output_dir (str): Directory to save upscaled images (optional)
            recursive (bool): Search subdirectories recursively
            quality (int): JPEG quality for output (1-100)
        
        Returns:
            list: Paths to upscaled images
        """
        input_dir = Path(input_dir)
        if not input_dir.exists() or not input_dir.is_dir():
            raise ValueError(f"Input directory not found: {input_dir}")
        
        # Set up output directory
        if output_dir is None:
            output_dir = input_dir / "upscaled_4x"
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
        
        print(f"📁 Found {len(image_files)} image(s) to process")
        
        # Process images with progress bar
        successful_outputs = []
        failed_files = []
        
        with tqdm(image_files, desc="🚀 Upscaling images", unit="img") as pbar:
            for img_file in pbar:
                try:
                    # Update progress bar description
                    pbar.set_postfix_str(f"Processing {img_file.name}")
                    
                    # Maintain directory structure
                    relative_path = img_file.relative_to(input_dir)
                    output_file = output_dir / relative_path.parent / f"{relative_path.stem}_upscaled_4x{relative_path.suffix}"
                    
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
    parser = argparse.ArgumentParser(description="Simple AI Image Upscaler - 4x Super Resolution")
    parser.add_argument("input", help="Input image file or directory")
    parser.add_argument("-o", "--output", help="Output file or directory path")
    parser.add_argument("-m", "--method", default="EDSR", 
                       choices=["EDSR", "ESPCN", "FSRCNN", "LapSRN"],
                       help="AI method to use for upscaling (default: EDSR)")
    parser.add_argument("-d", "--device", default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device to use for processing (default: auto)")
    parser.add_argument("-q", "--quality", type=int, default=95, metavar="1-100",
                       help="JPEG quality (1-100, default: 95)")
    parser.add_argument("-r", "--recursive", action="store_true",
                       help="Process subdirectories recursively (for directory input)")
    parser.add_argument("--batch", action="store_true",
                       help="Force batch processing mode")
    
    args = parser.parse_args()
    
    # Validate quality
    if not 1 <= args.quality <= 100:
        parser.error("Quality must be between 1 and 100")
    
    # Initialize upscaler
    print("🚀 Simple AI Image Upscaler - 4x Super Resolution")
    print("=" * 55)
    print(f"🔧 Method: {args.method}")
    print(f"💻 Device: {args.device}")
    print()
    
    upscaler = SimpleImageUpscaler(method=args.method, device=args.device)
    
    if not upscaler.initialize_model():
        print("\n❌ Failed to initialize model. Please check your OpenCV installation.")
        print("💡 Try: pip install opencv-contrib-python")
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