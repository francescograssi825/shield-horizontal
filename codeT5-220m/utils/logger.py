import torch
import os

def log_gpu_info():
    if torch.cuda.is_available():
        print(f"Available GPU: {torch.cuda.get_device_name()}")
        print(f"Free GPU memory: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print("Using CPU")
    
    # Optimize memory
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()