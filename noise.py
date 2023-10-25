import numpy as np

def get_noisy_image(img_np, sigma):
    Gauss_noise = np.random.normal(scale=sigma, size=img_np.shape)
    img_noisy_np = np.clip(img_np + Gauss_noise, 0, 1).astype(np.float32)

    return img_noisy_np