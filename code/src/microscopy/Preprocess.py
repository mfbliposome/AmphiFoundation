import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

from datetime import datetime
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

def transform_uint8(image_path):
    '''
    Transform a uint16 image to a uint8 image.

    Parameters:
    -----------
    image_path : str
        The path to the uint16 image file.

    Returns:
    --------
    image_uint8 : numpy.ndarray
        The transformed uint8 image.
    
    Notes:
    ------
    - The function reads an image in uint16 format, normalizes its pixel values to the range 0-255,
      and converts it to uint8 format.
    '''
    # Read the image as is (uint16)
    image_uint16 = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Normalize the image to the range 0-255
    image_normalized = cv2.normalize(image_uint16, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8
    image_uint8 = image_normalized.astype(np.uint8)

    return image_uint8

def Enhance_contrast(path, PlateName, image_type, sigma_size=50):
    '''
    Preprocess an image with background subtraction, contrast enhancement, and noise suppression.

    Parameters:
    -----------
    path : str
        The path to the image file.
    PlateName : str
        The name of the plate, used for saving results.
    image_type : str
        The type of the image. Supported types are 'uint16', 'gray', and 'RGB'.
    sigma_size : int, optional, default=50
        The sigma value for the Gaussian filter used in background subtraction.

    Returns:
    --------
    noise_suppressed : numpy.ndarray
        The preprocessed image after noise suppression.
    image : numpy.ndarray
        The original image loaded from the path.
    filename : str
        The filename of the processed image.

    Notes:
    ------
    - This function performs several preprocessing steps on the image:
      1. Background subtraction using a Gaussian filter.
      2. Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization).
      3. Noise suppression using a Gaussian blur.
    - The intermediate and final results are visualized and saved as an image file.
    '''
    # Load the TIFF image
    if image_type == 'uint16':
        image = transform_uint8(path)
    elif image_type == 'gray':
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif image_type == 'RGB':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError("Unsupported image type. Please specify either 'uint16', 'gray' or 'RGB'.")
    
    filename = os.path.basename(path)
    # Background subtraction
    background_gaussian = gaussian(image, sigma=sigma_size, preserve_range=True)
    image_gaussian = image - background_gaussian
    image_gaussian = np.clip(image_gaussian, 0, 255).astype(np.uint8)
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=5)
    image_clahe = clahe.apply(image_gaussian)
    # Noise suppression
    std_dev=0.5
    noise_suppressed = cv2.GaussianBlur(image_clahe, (0, 0), std_dev)

    fig, axs = plt.subplots(1, 5, figsize=(10,10))
    
    # Plot original image in the first subplot
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title("Original")
    axs[0].axis('off')
    
    # Plot background (gaussian) in the second subplot
    axs[1].imshow(background_gaussian, cmap='gray')
    axs[1].set_title("Background (gaussian)")
    axs[1].axis('off')
    
    # Plot background subtracted image in the third subplot
    axs[2].imshow(image_gaussian, cmap='gray')
    axs[2].set_title("Background subtracted")
    axs[2].axis('off')
    
    axs[3].imshow(image_clahe, cmap='gray')
    axs[3].set_title("CLAHE enhance")
    axs[3].axis('off')
    
    axs[4].imshow(noise_suppressed, cmap='gray')
    axs[4].set_title("Suppress noise")
    axs[4].axis('off')

    plt.tight_layout()
    plt.savefig(f'../../MicroscopyImage/VesicleDetection/Results_{PlateName}/'+f'{filename}_enhance_{formatted_datetime}.png')
    plt.close(fig)

    return noise_suppressed, image, filename


