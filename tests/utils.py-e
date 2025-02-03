"""
Utilities for running pytest tests in CytoDataFrame
"""

import base64
import re
from io import BytesIO
from typing import List, Tuple

import numpy as np
from PIL import Image

from cytodataframe import CytoDataFrame


def cytodataframe_image_display_contains_green_pixels(
    frame: CytoDataFrame, image_cols: List[str]
) -> bool:
    """
    Determines if relevant image from the CytoDataFrame HTML
    contains green pixels.

    Args:
        frame (CytoDataFrame):
            A custom `CytoDataFrame` object which includes image paths.
        image_cols (List[str]):
            A list of column names in the `CytoDataFrame`
            that contain images paths.

    Returns:
        bool:
            True if any greenish pixels are found in relevant
            image within the HTML, otherwise False.

    Raises:
        ValueError:
            If no base64-encoded image data is found in the
            HTML representation of the given columns.
    """

    # gather HTML output from CytoDataFrame
    html_output = frame[image_cols]._repr_html_()

    # Extract all base64 image data from the HTML
    matches = re.findall(r'data:image/png;base64,([^"]+)', html_output)

    # check that we have matches
    if not len(matches) > 0:
        raise ValueError("No base64 image data found in HTML")

    # Select the third base64 image data (indexing starts from 0)
    # (we expect the first ones to not contain outlines based on the
    # html and example data)
    base64_data = matches[2]

    # Decode the base64 image data
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Check for the presence of green pixels in the image
    image_array = np.array(image)

    # gather color channels from image
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Define a threshold to identify greenish pixels
    green_threshold = 50
    green_pixels = (
        (green_channel > green_threshold)
        & (green_channel > red_channel)
        & (green_channel > blue_channel)
    )

    # return true/false if there's at least one greenish pixel in the image
    return np.any(green_pixels)


def create_sample_image(
    width: int, height: int, color: Tuple[int, int, int, int] = (0, 0, 0, 255)
) -> Image:
    """
    Creates a sample RGBA image with the specified dimensions
    and background color.

    Args:
        width (int):
            The width of the image.
        height (int):
            The height of the image.
        color (Tuple[int, int, int, int]):
            The background color of the image, represented as an
            RGBA tuple (default is black).

    Returns:
        PIL.Image.Image:
            A PIL Image object with the specified color and dimensions.
    """
    image = Image.new("RGBA", (width, height), color)
    return image


def create_sample_outline(
    width: int, height: int, outline_color: Tuple[int, int, int] = (255, 0, 0)
) -> Image:
    """
    Creates a sample outline image with a red outline
    drawn on a transparent background.

    Args:
        width (int):
            The width of the outline image.
        height (int):
            The height of the outline image.
        outline_color (Tuple[int, int, int]):
            The color of the outline in RGB format (default is red).

    Returns:
        PIL.Image.Image:
            A PIL Image object with a red outline on a transparent background.
    """
    # Transparent background
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Create a simple outline shape
    for x in range(10, 100):
        for y in range(10, 100):
            # A simple border outline
            if x == 10 or y == 10 or x == 99 or y == 99:
                # Red outline
                image.putpixel((x, y), (*outline_color, 255))
    return image
