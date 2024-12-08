"""
Helper functions for working with images in the context of CytoDataFrames.
"""

import cv2
import numpy as np
import skimage
import skimage.io
import skimage.measure
from PIL import Image, ImageEnhance
from skimage import draw, exposure
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


def draw_outline_on_image_from_outline(
    orig_image: np.ndarray, outline_image_path: str
) -> np.ndarray:
    """
    Draws green outlines on an image based on an outline image and returns
    the combined result.
    """

    # Load the outline image
    outline_image = skimage.io.imread(outline_image_path)

    # Resize if necessary
    if outline_image.shape[:2] != orig_image.shape[:2]:
        outline_image = skimage.transform.resize(
            outline_image,
            orig_image.shape[:2],
            preserve_range=True,
            anti_aliasing=True,
        ).astype(orig_image.dtype)

    # Create a mask for non-black areas (with threshold)
    threshold = 10  # Adjust as needed
    # Grayscale
    if outline_image.ndim == 2:  # noqa: PLR2004
        non_black_mask = outline_image > threshold
    else:  # RGB/RGBA
        non_black_mask = np.any(outline_image[..., :3] > threshold, axis=-1)

    # Ensure the original image is RGB
    if orig_image.ndim == 2:  # noqa: PLR2004
        orig_image = np.stack([orig_image] * 3, axis=-1)
    elif orig_image.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError("Original image must have 3 channels (RGB).")

    # Ensure uint8 data type
    if orig_image.dtype != np.uint8:
        orig_image = (orig_image * 255).astype(np.uint8)

    # Apply the green outline
    combined_image = orig_image.copy()
    combined_image[non_black_mask] = [0, 255, 0]  # Green in uint8

    return combined_image


def draw_outline_on_image_from_mask(
    orig_image: np.ndarray, mask_image_path: str
) -> np.ndarray:
    """
    Draws green outlines on an image based on a binary mask and returns
    the combined result.

    Args:
        orig_image (np.ndarray):
            Image which a mask will be applied to. Must be a NumPy array.
        mask_image_path (str):
            Path to the binary mask image file.

    Returns:
        np.ndarray:
            The resulting image with the green outline applied.
    """
    # Load the binary mask image
    mask_image = skimage.io.imread(mask_image_path)

    # Ensure the original image is RGB
    # Grayscale input
    if orig_image.ndim == 2:  # noqa: PLR2004
        orig_image = np.stack([orig_image] * 3, axis=-1)
    # Unsupported input
    elif orig_image.shape[-1] != 3:  # noqa: PLR2004
        raise ValueError("Original image must have 3 channels (RGB).")

    # Ensure the mask is 2D (binary)
    if mask_image.ndim > 2:  # noqa: PLR2004
        mask_image = mask_image[..., 0]  # Take the first channel if multi-channel

    # Detect contours from the mask
    contours = skimage.measure.find_contours(mask_image, level=0.5)

    # Create an outline image with the same shape as the original image
    outline_image = np.zeros_like(orig_image)

    # Draw contours as green lines
    for contour in contours:
        rr, cc = draw.polygon_perimeter(
            np.round(contour[:, 0]).astype(int),
            np.round(contour[:, 1]).astype(int),
            shape=orig_image.shape[:2],
        )
        # Assign green color to the outline in all three channels
        outline_image[rr, cc, :] = [0, 255, 0]

    # Combine the original image with the green outline
    combined_image = orig_image.copy()
    mask = np.any(outline_image > 0, axis=-1)  # Non-zero pixels in the outline
    combined_image[mask] = outline_image[mask]

    return combined_image


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
    # RGBA image
    if image_np.shape[-1] == 4:  # noqa: PLR2004
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

        # Convert processed RGB back to 8-bit
        equalized_rgb_np = img_as_ubyte(equalized_rgb_np)

        # Combine the processed RGB with the original alpha channel
        final_image_np = np.dstack([equalized_rgb_np, alpha_np])

    # Grayscale image
    elif len(image_np.shape) == 2:  # noqa: PLR2004
        # Apply CLAHE directly to the grayscale image
        final_image_np = exposure.equalize_adapthist(
            image_np,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        )
        # Convert processed image back to 8-bit
        final_image_np = img_as_ubyte(final_image_np)

    # RGB image
    elif image_np.shape[-1] == 3:  # noqa: PLR2004
        # Placeholder for processed RGB channels
        equalized_rgb_np = np.zeros_like(image_np, dtype=np.float32)

        # Apply AHE to each RGB channel separately
        for channel in range(3):
            equalized_rgb_np[:, :, channel] = exposure.equalize_adapthist(
                image_np[:, :, channel],
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=nbins,
            )

        # Convert processed RGB back to 8-bit
        final_image_np = img_as_ubyte(equalized_rgb_np)

    else:
        raise ValueError(
            "Unsupported image format. Ensure the image is grayscale, RGB, or RGBA."
        )

    # Convert NumPy array back to PIL Image
    return Image.fromarray(final_image_np)
