import os
import urllib.request
import hashlib
from pathlib import Path

def download_beats_model():
    """Download the BEATs model with verification"""
    model_url = "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
    model_path = Path("./BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt")
    
    if model_path.exists():
        print(f"Model already exists at {model_path}")
        return str(model_path)
    
    print("Downloading BEATs model...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Model downloaded to {model_path}")
        
        # Verify file size (should be around 95MB)
        file_size = model_path.stat().st_size / (1024 * 1024)  # MB
        if file_size < 90:  # Less than 90MB indicates incomplete download
            print(f"Warning: Downloaded file seems too small ({file_size:.1f}MB)")
            model_path.unlink()  # Delete incomplete file
            raise RuntimeError("Download appears incomplete")
            
    except Exception as e:
        print(f"Failed to download model: {e}")
        if model_path.exists():
            model_path.unlink()
        raise
    
    return str(model_path)

if __name__ == "__main__":
    download_beats_model()
