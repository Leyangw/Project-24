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
    noise = torch.randn(tensor.size()) * std_dev + mean
    noisy_tensor = tensor + noise
    return noisy_tensor

# Example usage with a PyTorch tensor
# Create a tensor with random values (replace this with your own tensor)
if __name__ == '__main__':
    input_tensor = torch.randn(3, 512, 512)

# Add Gaussian noise to the tensor
    noisy_tensor = add_gaussian_noise(input_tensor, mean=0, std_dev=0.1)
