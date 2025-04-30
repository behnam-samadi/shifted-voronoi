import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)


import torch
print("Compiled with CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Current GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

