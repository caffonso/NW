import numpy as np

def normalize_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)
