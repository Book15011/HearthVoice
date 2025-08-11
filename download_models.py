import os
import sys
import argparse
import subprocess
from urllib.parse import urlparse

# --- Helper Functions ---

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    from shutil import which
    return which(name) is not None

def run_command(command, description):
    """Runs a command and prints success or failure."""
    print(f"--- {description} ---")
    try:
        # Using shell=True for simplicity with complex commands.
        subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ Success: {description} completed.\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description} failed. Command '{e.cmd}' returned non-zero exit status {e.returncode}.")
        if e.stdout:
            print(f"   stdout: {e.stdout.decode().strip()}")
        if e.stderr:
            print(f"   stderr: {e.stderr.decode().strip()}")
        print("   Please check the error message above and try again.\n")
        return False
    except FileNotFoundError:
        tool = command.split()[0]
        print(f"❌ Error: Command '{tool}' not found. Is it installed and in your PATH?")
        return False

def download_file(url, dest_path):
    """Downloads a file using requests with a progress bar."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError:
        print("Required packages `requests` and `tqdm` not found.")
        print("Please install them with: pip install requests tqdm")
        sys.exit(1)

    filename = os.path.basename(urlparse(url).path)
    dest_file = os.path.join(dest_path, filename)
    os.makedirs(dest_path, exist_ok=True)

    if os.path.exists(dest_file):
        print(f"✅ File already exists: {dest_file}")
        return

    print(f"Downloading {filename} to {dest_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(dest_file, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc=filename
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print(f"✅ Successfully downloaded {filename}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to download {url}. Error: {e}")
        if os.path.exists(dest_file):
            os.remove(dest_file) # Clean up partial download

# --- Model Download Functions ---

def download_cosyvoice_models():
    """Downloads CosyVoice models from ModelScope."""
    print("--- Downloading CosyVoice Models ---")
    try:
        from modelscope.hub.snapshot_download import snapshot_download
    except ImportError:
        print("Required package `modelscope` not found.")
        print("Please install it with: pip install modelscope")
        return

    models = {
        "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
        "CosyVoice-300M": "iic/CosyVoice-300M",
        "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
        "CosyVoice-300M-Instruct": "iic/CosyVoice-300M-Instruct",
        "CosyVoice-ttsfrd": "iic/CosyVoice-ttsfrd"
    }
    
    for name, model_id in models.items():
        print(f"\nDownloading {name}...")
        dest_path = os.path.join("pretrained_models", name)
        if os.path.exists(dest_path) and os.listdir(dest_path):
             print(f"✅ Model '{name}' already exists in {dest_path}. Skipping.")
             continue
        try:
            # ModelScope's download function manages the actual destination within the cache_dir
            snapshot_download(model_id, cache_dir="pretrained_models", local_dir=name)
            print(f"✅ Successfully downloaded {name} to {dest_path}")
        except Exception as e:
            print(f"❌ Failed to download {name}. Error: {e}")
            print("   Please check your network connection and if you have `modelscope` installed correctly.")

def download_sadtalker_models():
    """Downloads SadTalker checkpoints and dependencies."""
    print("\n--- Downloading SadTalker Models ---")
    
    if not is_tool('git'):
        print("❌ Git is not installed. Please install it to download SadTalker dependencies.")
        return

    sadtalker_path = os.path.join("third_party", "SadTalker")
    checkpoints_path = os.path.join(sadtalker_path, "checkpoints")
    gfpgan_weights_path = os.path.join(sadtalker_path, "gfpgan", "weights")

    if not os.path.exists(sadtalker_path):
        print(f"Cloning SadTalker repository into {sadtalker_path}...")
        run_command(
            f"git clone https://github.com/OpenTalker/SadTalker.git {sadtalker_path}",
            "Cloning SadTalker"
        )
    else:
        print(f"✅ SadTalker repository already exists at {sadtalker_path}.")

    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(gfpgan_weights_path, exist_ok=True)

    sadtalker_files = {
        "checkpoints": [
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/checkpoints/SadTalker_V0.0.2_256.safetensors",
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/checkpoints/mapping_00109-model.pth.tar",
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/checkpoints/wav2lip.pth"
        ],
        "gfpgan_weights": [
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/gfpgan/weights/GFPGANv1.4.pth",
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/gfpgan/weights/detection_Resnet50_Final.pth",
            "https://hf-mirror.com/vinthony/SadTalker/resolve/main/gfpgan/weights/parsing_parsenet.pth"
        ]
    }

    print("\nDownloading SadTalker checkpoint files...")
    for url in sadtalker_files["checkpoints"]:
        download_file(url, checkpoints_path)
        
    print("\nDownloading GFPGAN weight files for SadTalker...")
    for url in sadtalker_files["gfpgan_weights"]:
        download_file(url, gfpgan_weights_path)

def precache_aux_models():
    """Downloads and caches Whisper and Arabic TTS models."""
    print("\n--- Pre-caching Auxiliary Models (Whisper and Arabic TTS) ---")
    print("This will download models to your user cache directory (e.g., ~/.cache).")

    print("\nDownloading Whisper 'base' model...")
    try:
        import whisper
        whisper.load_model("base")
        print("✅ Whisper model cached successfully.")
    except ImportError:
        print("`openai-whisper` not installed. Skipping. To enable, run: pip install openai-whisper")
    except Exception as e:
        print(f"❌ Failed to cache Whisper model. Error: {e}")

    print("\nDownloading Arabic TTS model (facebook/mms-tts-ara)...")
    try:
        from transformers import AutoProcessor, AutoModel
        AutoProcessor.from_pretrained("facebook/mms-tts-ara")
        AutoModel.from_pretrained("facebook/mms-tts-ara")
        print("✅ Arabic TTS model cached successfully.")
    except ImportError:
        print("`transformers` not installed. Skipping. To enable, run: pip install transformers sentencepiece")
    except Exception as e:
        print(f"❌ Failed to cache Arabic TTS model. Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Download all necessary models for the ElderCompanion AI project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--all', action='store_true', help="Download all required models.")
    parser.add_argument('--cosyvoice', action='store_true', help="Download only CosyVoice models.")
    parser.add_argument('--sadtalker', action='store_true', help="Download only SadTalker models.")
    parser.add_argument('--aux', action='store_true', help="Pre-cache only auxiliary models (Whisper, etc.).")

    args = parser.parse_args()

    if not any([args.cosyvoice, args.sadtalker, args.aux, args.all]):
        parser.print_help()
        print("\nError: Please specify which models to download, or use --all.")
        sys.exit(1)

    if args.cosyvoice or args.all:
        download_cosyvoice_models()

    if args.sadtalker or args.all:
        download_sadtalker_models()

    if args.aux or args.all:
        precache_aux_models()
        
    print("\n🎉 All selected downloads are complete!")

if __name__ == "__main__":
    main()