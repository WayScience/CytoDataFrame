"""
Utilities for running pytest tests in CytoDataFrame
"""

import base64
import re
from io import BytesIO
from typing import List

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
