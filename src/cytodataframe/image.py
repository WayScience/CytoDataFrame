"""
Helper functions for working with images in the context of CytoDataFrames.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure
from skimage.filters import gaussian
from skimage.util import img_as_ubyte


def is_image_too_dark(
    image: Image.Image, pixel_brightness_threshold: float = 10.0
) -> bool:
    """
    Check if the image is too dark based on the mean brightness.
    By "too dark" we mean not as visible to the human eye.

    Args:
        image (Image):
            The input PIL Image.
        threshold (float):
            The brightness threshold below which the image is considered too dark.

    Returns:
        bool:
            True if the image is too dark, False otherwise.
    """
    # Convert the image to a numpy array and then to grayscale
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Calculate the mean brightness
    mean_brightness = np.mean(gray_image)

    return mean_brightness < pixel_brightness_threshold


def adjust_image_brightness(image: Image.Image) -> Image.Image:
    """
    Adjust the brightness of an image using histogram equalization.

    Args:
        image (Image):
            The input PIL Image.

    Returns:
        Image:
            The brightness-adjusted PIL Image.
    """
    # Convert the image to numpy array and then to grayscale
    img_array = np.array(image)
    gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)

    # Apply histogram equalization to improve the contrast
    equalized_image = cv2.equalizeHist(gray_image)

    # Convert back to RGBA
    img_array[:, :, 0] = equalized_image  # Update only the R channel
    img_array[:, :, 1] = equalized_image  # Update only the G channel
    img_array[:, :, 2] = equalized_image  # Update only the B channel

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(img_array)

    # Slightly reduce the brightness
    enhancer = ImageEnhance.Brightness(enhanced_image)
    reduced_brightness_image = enhancer.enhance(0.7)

    return reduced_brightness_image


def adjust_with_adaptive_histogram_equalization(image: Image.Image) -> Image.Image:
    """
    Adaptive histogram equalization with additional smoothing to reduce graininess.

    Parameters:
    image (Image): A PIL Image to be processed.

    Returns:
    Image: A PIL Image after applying adaptive histogram equalization.
    """
    # Convert PIL Image to NumPy array
    image_np = np.asarray(image)

    # Adjust parameters dynamically
    kernel_size = (
        image_np.shape[0] // 10,
        image_np.shape[1] // 10,
    )  # Larger regions for smoother adjustment
    clip_limit = 0.02  # Lower clip limit to suppress over-enhancement
    nbins = 512  # Increase bins for finer histogram granularity

    # Check if the image has an alpha channel (RGBA)
    if image_np.shape[-1] == 4:  # RGBA image
        # Split the channels: RGB and A
        rgb_np = image_np[:, :, :3]
        alpha_np = image_np[:, :, 3]

        # Placeholder for processed RGB channels
        equalized_rgb_np = np.zeros_like(rgb_np, dtype=np.float32)

        # Apply AHE to each RGB channel separately
        for channel in range(3):  # Only process R, G, and B channels
            equalized_rgb_np[:, :, channel] = exposure.equalize_adapthist(
                rgb_np[:, :, channel],
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=nbins,
            )

        # Apply Gaussian smoothing to reduce graininess
        equalized_rgb_np = gaussian(equalized_rgb_np, sigma=0.5, channel_axis=-1)

        # Convert processed RGB back to 8-bit
        equalized_rgb_np = img_as_ubyte(equalized_rgb_np)

        # Combine the processed RGB with the original alpha channel
        final_image_np = np.dstack([equalized_rgb_np, alpha_np])

    elif len(image_np.shape) == 2:  # Grayscale
        # Apply CLAHE directly to the grayscale image
        final_image_np = exposure.equalize_adapthist(
            image_np,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )
        # Apply Gaussian smoothing to reduce graininess
        final_image_np = gaussian(final_image_np, sigma=0.5)
        # Convert processed image back to 8-bit
        final_image_np = img_as_ubyte(final_image_np)

    else:
        raise ValueError(
            "Unsupported image format. Ensure the image is grayscale or RGBA."
        )

    # Convert NumPy array back to PIL Image
    return Image.fromarray(final_image_np)
