
import torch
import sys

print(f"--- PyTorch/CUDA Diagnostic ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if "+cu" not in torch.__version__ and "cuda" not in torch.__version__:
    print("\nWARNING: You have a CPU-only version of PyTorch installed. Please reinstall with CUDA support.")
    print("This is often caused by a conflicting `pip` installation. Try running `pip uninstall torch`.")

if not torch.cuda.is_available():
    print("\nCUDA is not available to PyTorch. This is likely a driver or installation issue.")
    # Add a hint about the NVML error
    try:
        # This will likely fail if the driver is broken
        torch.cuda.init()
    except Exception as e:
        if "Can't initialize NVML" in str(e) or "could not be found" in str(e):
             print("Error during CUDA initialization:", str(e))
             print("This confirms an NVIDIA driver problem. Please run `nvidia-smi` in your terminal.")
             print("If that command fails, you must reinstall your NVIDIA drivers.")
else:
    try:
        print(f"CUDA version built with PyTorch: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"Number of CUDA devices: {device_count}")
        if device_count > 0:
            for i in range(device_count):
                print(f"--- Device {i} ---")
                print(f"Name: {torch.cuda.get_device_name(i)}")
                print(f"CUDA Capability: {torch.cuda.get_device_capability(i)}")
                # Let's try allocating a tensor to be sure
                print("Attempting to allocate a tensor on this device...")
                a = torch.tensor([1.0]).to(f"cuda:{i}")
                print("Tensor allocation successful.")
                del a
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nAn error occurred during CUDA diagnostics: {e}")
        print("This could indicate a problem with the driver or the PyTorch installation.")
print(f"-----------------------------")
