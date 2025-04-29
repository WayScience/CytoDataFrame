"""
Utilities for running pytest tests in CytoDataFrame
"""

import base64
import re
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from cytodataframe import CytoDataFrame


def cytodataframe_image_display_contains_pixels(
    frame: CytoDataFrame, image_cols: List[str], color_conditions: Dict
) -> bool:
    """
    Determines if relevant images from the CytoDataFrame HTML
    contain pixels matching specified color conditions.

    Args:
        frame (CytoDataFrame):
            A custom `CytoDataFrame` object which includes image paths.
        image_cols (List[str]):
            A list of column names in the `CytoDataFrame`
            that contain image paths.
        color_conditions (dict):
            A list of dictionaries specifying color conditions.
            Each dictionary should have keys 'red', 'green', and 'blue',
            with corresponding threshold values. If a threshold is `None`,
            that channel is ignored.

            Examples:
            {"red": 50, "green": 100, "blue": None},  # Greenish pixels
            {"red": 200, "green": None, "blue": 50},  # Reddish pixels

    Returns:
        bool:
            True if any pixels matching the specified color conditions
            are found in the relevant images within the HTML, otherwise False.

    Raises:
        ValueError:
            If no base64-encoded image data is found in the
            HTML representation of the given columns.
    """

    # Gather HTML output from CytoDataFrame
    html_output = frame[image_cols]._repr_html_()

    # Extract all base64 image data from the HTML
    matches = re.findall(r'data:image/png;base64,([^"]+)', html_output)

    # Check that we have matches
    if not matches:
        raise ValueError("No base64 image data found in HTML")

    # Select the third base64 image data (indexing starts from 0)
    # (we expect the first ones to not contain outlines based on the
    # HTML and example data)
    base64_data = matches[2]

    # Decode the base64 image data
    image_data = base64.b64decode(base64_data)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Gather color channels from the image
    red_channel = image_array[:, :, 0]
    green_channel = image_array[:, :, 1]
    blue_channel = image_array[:, :, 2]

    # Check for the presence of pixels matching any of the color conditions
    red_threshold = color_conditions.get("red")
    green_threshold = color_conditions.get("green")
    blue_threshold = color_conditions.get("blue")

    # Start with a mask of all True (all pixels are initially valid)
    matching_pixels = np.ones_like(red_channel, dtype=bool)

    # Apply thresholds only if they are not None
    if red_threshold is not None:
        matching_pixels &= red_channel >= red_threshold
    if green_threshold is not None:
        matching_pixels &= green_channel >= green_threshold
    if blue_threshold is not None:
        matching_pixels &= blue_channel >= blue_threshold

    # If any matching pixels are found, return True
    # If no matching pixels are found for any condition, return False
    return np.any(matching_pixels)


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
