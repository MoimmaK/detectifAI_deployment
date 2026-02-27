"""
Report Generation Module Setup Script

Run this script to:
1. Check/install dependencies
2. Download the LLM model
3. Create necessary directories
4. Verify the installation
"""

import os
import sys
import subprocess


def check_python_version():
    """Check Python version."""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"❌ Python 3.9+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing Python dependencies...")
    
    packages = [
        "llama-cpp-python",
        "huggingface_hub",
        "jinja2",
        "markdown",
        "Pillow",
        "reportlab",
    ]
    
    # Optional packages (may fail on some systems)
    optional_packages = [
        "weasyprint",  # Requires GTK3 on Windows
    ]
    
    # Install required packages
    for package in packages:
        print(f"  Installing {package}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ❌ Failed to install {package}")
            return False
    
    # Try optional packages
    for package in optional_packages:
        print(f"  Installing {package} (optional)...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package, "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError:
            print(f"  ⚠️ {package} not installed (PDF export may not work)")
    
    return True


def download_model():
    """Download the LLM model."""
    print("\n🤖 Downloading LLM model...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        model_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        # Download Qwen2.5-3B-Instruct Q4 quantized
        print("  Downloading Qwen2.5-3B-Instruct (Q4_K_M, ~2GB)...")
        print("  This may take several minutes depending on your connection...")
        
        downloaded_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"  ✅ Model downloaded to: {downloaded_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to download model: {e}")
        print("  You can manually download from:")
        print("  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    base_dir = os.path.dirname(__file__)
    directories = [
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'templates'),
        os.path.join(base_dir, 'prompts'),
        os.path.join(os.path.dirname(base_dir), 'video_processing_outputs', 'reports'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✅ {directory}")
    
    return True


def verify_installation():
    """Verify the installation."""
    print("\n🔬 Verifying installation...")
    
    # Check imports
    modules = [
        ('llama_cpp', 'llama-cpp-python'),
        ('huggingface_hub', 'huggingface_hub'),
        ('jinja2', 'Jinja2'),
        ('markdown', 'markdown'),
        ('PIL', 'Pillow'),
        ('reportlab', 'reportlab'),
    ]
    
    all_ok = True
    for module, package in modules:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} not found")
            all_ok = False
    
    # Check weasyprint (optional)
    try:
        import weasyprint
        print("  ✅ weasyprint (PDF export available)")
    except (ImportError, OSError):
        print("  ⚠️ weasyprint not available (PDF export disabled, use HTML)")
    
    # Check model file
    model_path = os.path.join(
        os.path.dirname(__file__), 
        'models', 
        'qwen2.5-3b-instruct-q4_k_m.gguf'
    )
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✅ LLM model found ({size_mb:.0f} MB)")
    else:
        print("  ⚠️ LLM model not found - will download on first use")
    
    return all_ok


def test_generation():
    """Test the report generation system."""
    print("\n🧪 Testing report generation...")
    
    try:
        from report_generation import ReportGenerator
        
        generator = ReportGenerator()
        print("  ✅ ReportGenerator initialized")
        
        # Test without actual data
        print("  ✅ Module imports successful")
        print("\n  Note: Full test requires a processed video in the database")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


def main():
    """Run the setup process."""
    print("=" * 60)
    print("🛡️ DetectifAI Report Generation Module Setup")
    print("=" * 60)
    
    steps = [
        ("Check Python version", check_python_version),
        ("Create directories", create_directories),
        ("Install dependencies", install_dependencies),
        ("Verify installation", verify_installation),
    ]
    
    all_passed = True
    for step_name, step_func in steps:
        if not step_func():
            all_passed = False
            print(f"\n⚠️ Step '{step_name}' had issues. Continuing...")
    
    # Ask about model download
    print("\n" + "=" * 60)
    print("📥 Model Download")
    print("=" * 60)
    print("\nThe LLM model (~2GB) is required for AI-generated report content.")
    print("It will be automatically downloaded on first use, or you can download now.")
    
    response = input("\nDownload model now? [y/N]: ").strip().lower()
    if response == 'y':
        download_model()
    else:
        print("Skipping model download. Will download on first use.")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 Setup Summary")
    print("=" * 60)
    
    if all_passed:
        print("\n✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Process a video through DetectifAI pipeline")
        print("2. Generate a report:")
        print("   from report_generation import ReportGenerator")
        print("   generator = ReportGenerator()")
        print("   report = generator.generate_report('video_id_here')")
        print("   generator.export_html(report)")
    else:
        print("\n⚠️ Setup completed with some warnings.")
        print("Check the messages above for details.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
