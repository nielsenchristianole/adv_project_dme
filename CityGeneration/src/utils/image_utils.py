
import numpy as np
from PIL import Image

def pad_to_square(image : np.ndarray, pad_value = 0) -> np.ndarray:
    """
    Pad a numpy array into into square
    """
    
    height, width = image.shape[:2]
    if height == width:
        return image
    
    size = max(height, width)
    padded_image = np.ones((size, size) + image.shape[2:], dtype=image.dtype)
    padded_image = padded_image * pad_value
    
    y_offset = (size - height) // 2
    x_offset = (size - width) // 2
    
    padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = image
    return padded_image
    
def crop_to_square(image: np.ndarray) -> np.ndarray:
    """
    Crops an image into square 
    """
    height, width = image.shape[:2]
    if height == width:
        return image
    
    size = min(height, width)
    y_offset = (height - size) // 2
    x_offset = (width - size) // 2
    
    cropped_image = image[y_offset:y_offset + size, x_offset:x_offset + size]
    return cropped_image

def resize(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    Resize an image to the given size.
    """
    resized = np.array(Image.fromarray(image).resize(size))
    return resized