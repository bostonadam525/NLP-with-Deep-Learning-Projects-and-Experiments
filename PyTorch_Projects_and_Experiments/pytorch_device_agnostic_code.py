# 1. Check for mac or windows and then check for GPU/MPS or CPU
import torch
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device

# 2. standard device agnostic code
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
device

# 3. Another approach to check for Apple vs. NVIDIA GPUs
if torch.cuda.is_available():
    device = "cuda" # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps" # Apple GPU
else:
    device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

print(f"Using device: {device}")


## 4. Multiple GPU support with CUDA 
##This approach ensures that the code runs on the appropriate device without requiring modifications. 
## When working with multiple GPUs, `torch.nn.DataParallel` or `torch.distributed.DataParallel` can be used to parallelize 
##the computation across the available GPUs.

# Move model to all available GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.to(device)
