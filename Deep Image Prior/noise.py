import torch

def add_gaussian_noise(tensor, mean=0, std_dev=1):
    """
    Add Gaussian noise to a PyTorch tensor.

    Parameters:
    - tensor: Input PyTorch tensor.
    - mean: Mean of the Gaussian noise.
    - std_dev: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy tensor.
    """
    noise = torch.randn(tensor.size()) * std_dev/255 + mean/255
    noisy_tensor = tensor + noise
    return noisy_tensor

