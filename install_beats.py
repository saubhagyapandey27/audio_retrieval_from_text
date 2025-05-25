import subprocess
import sys
import os

def install_beats_dependencies():
    """Install BEATs dependencies with proper error handling"""
    
    print("Installing BEATs dependencies...")
    
    # Install fairseq with no-build-isolation to avoid PyTorch conflicts
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "--no-build-isolation",
            "fairseq==0.12.2"
        ])
        print("✓ fairseq installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install fairseq")
        print("Trying alternative installation...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-deps",
                "fairseq==0.12.2"
            ])
            print("✓ fairseq installed with --no-deps")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install fairseq: {e}")
            return False
    
    return True

if __name__ == "__main__":
    if install_beats_dependencies():
        print("All dependencies installed successfully!")
        # Download model
        from setup_beats import download_beats_model
        download_beats_model()
    else:
        print("Failed to install dependencies")
        sys.exit(1)