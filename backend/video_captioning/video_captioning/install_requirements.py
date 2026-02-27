"""
Install minimal requirements for video captioning module
"""

import subprocess
import sys
import os


def install_package(package):
    """Install a package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False


def check_package(package):
    """Check if a package is already installed"""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


def main():
    """Install required packages"""
    print("Checking and installing requirements for video captioning module...")
    
    # Essential packages for testing
    packages = [
        "torch",
        "torchvision", 
        "transformers",
        "sentence-transformers",
        "Pillow",
        "opencv-python",
        "numpy"
    ]
    
    # Check what's already installed
    installed = []
    to_install = []
    
    for package in packages:
        # Map package names to import names
        import_name = package
        if package == "opencv-python":
            import_name = "cv2"
        elif package == "Pillow":
            import_name = "PIL"
        
        if check_package(import_name):
            installed.append(package)
            print(f"✓ {package} already installed")
        else:
            to_install.append(package)
    
    if not to_install:
        print("\nAll required packages are already installed!")
        return
    
    print(f"\nNeed to install: {', '.join(to_install)}")
    
    # Install missing packages
    failed = []
    for package in to_install:
        if not install_package(package):
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  Failed to install: {', '.join(failed)}")
        print("You may need to install these manually or check your internet connection.")
    else:
        print("\n🎉 All packages installed successfully!")
        print("You can now run the test with: python test_runner.py")


if __name__ == "__main__":
    main()