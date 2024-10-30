"""
Helper functions for working with images in the context of CytoDataFrames.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def is_image_too_dark(image: Image, pixel_brightness_threshold: float = 10.0) -> bool:
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


def adjust_image_brightness(image: Image) -> Image:
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
