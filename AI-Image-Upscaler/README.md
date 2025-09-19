# 🚀 AI Image Upscaler

A powerful AI-driven image upscaling tool that enhances your images by 4x their original resolution using state-of-the-art deep learning models.

## ✨ Features

- **4x Image Upscaling**: Increase image resolution by 4 times using advanced AI
- **Multiple AI Models**: Choose from different models optimized for various image types
- **Batch Processing**: Process entire directories of images at once
- **Fast Processing**: Optimized for speed with ESPCN model (sub-second processing)
- **Multiple Formats**: Support for JPG, PNG, BMP, TIFF, and WebP formats
- **Easy Setup**: One-click installation with automated dependency management

## 🎯 Supported AI Models

- **EDSR** - Best quality, slower processing (~38MB model)
- **ESPCN** - Fastest processing, good quality (~100KB model) - **Recommended**
- **FSRCNN** - Balanced speed and quality
- **LapSRN** - Good for detailed images

## ⚡ Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/AI-Image-Upscaler.git
   cd AI-Image-Upscaler
   ```

2. **Run setup (Windows):**
   ```bash
   python setup.py
   ```
   Or double-click `setup.bat`

3. **Activate environment:**
   ```bash
   ai_upscaler_env\Scripts\Activate.ps1  # PowerShell
   # or
   ai_upscaler_env\Scripts\activate.bat  # Command Prompt
   ```

### Usage

```bash
# Basic usage - upscale a single image
python simple_upscaler.py "your_image.jpg"

# Fast processing with ESPCN model (recommended)
python simple_upscaler.py "your_image.jpg" -m ESPCN

# Batch process a folder
python simple_upscaler.py "image_folder" -m ESPCN

# Custom output path
python simple_upscaler.py "input.jpg" -o "output_4k.jpg" -m ESPCN
```

## 📊 Performance

| Model | Speed (900x1200 → 3600x4800) | Quality | Recommended Use |
|-------|------------------------------|---------|-----------------|
| **ESPCN** | ~0.5 seconds | Good | General use, batch processing |
| **EDSR** | ~4 minutes | Excellent | High-quality single images |
| **FSRCNN** | ~30 seconds | Very Good | Balanced processing |

## 🛠️ Requirements

- Python 3.7 or higher
- OpenCV with contrib modules
- 8GB+ RAM recommended
- Windows 10/11, macOS, or Linux

## 📁 Project Structure

```
AI-Image-Upscaler/
├── simple_upscaler.py    # Main AI upscaler program
├── setup.py              # Installation script
├── setup.bat             # Windows setup shortcut
├── requirements.txt      # Dependencies
├── README.md             # This guide
└── models/               # AI models (auto-downloaded)
    ├── EDSR_x4.pb        # High quality model
    └── ESPCN_x4.pb       # Fast processing model
```

## 🎨 Examples

Transform your images from low resolution to high resolution:
- 512×512 → 2048×2048 (4x larger)
- 1920×1080 → 7680×4320 (4K to 8K equivalent)
- Any resolution → 4x width and height

## 🔧 Command Line Options

```bash
python simple_upscaler.py [-h] [-o OUTPUT] [-m {EDSR,ESPCN,FSRCNN,LapSRN}] 
                          [-d {auto,cuda,cpu}] [-q 1-100] [-r] [--batch] input

Options:
  -m, --method    AI method to use (default: EDSR)
  -o, --output    Output file or directory path
  -d, --device    Device to use: auto, cuda, cpu
  -q, --quality   JPEG quality 1-100 (default: 95)
  -r, --recursive Process subdirectories recursively
  --batch         Force batch processing mode
```

## 💡 Tips

1. **For Speed**: Use ESPCN model (`-m ESPCN`)
2. **For Quality**: Use EDSR model (`-m EDSR`) 
3. **For Balance**: Use FSRCNN model (`-m FSRCNN`)
4. **Large Images**: Models automatically handle memory optimization
5. **Batch Processing**: Most efficient for multiple images

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

Built using:
- [OpenCV DNN Super Resolution](https://docs.opencv.org/4.x/d4/d69/tutorial_dnn_superres_tutorial.html)
- Pre-trained models from various research papers
- Community contributions and feedback

---

**Happy Upscaling! 🎉**