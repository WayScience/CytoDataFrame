"""
Tests cosmicqc image module
"""

import pathlib
from typing import Tuple

import numpy as np
import pytest
from PIL import Image
from skimage.draw import disk
from skimage.io import imsave

from cytodataframe.image import (
    adjust_image_brightness,
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
    is_image_too_dark,
)


def test_is_image_too_dark_with_dark_image(fixture_dark_image: Image):
    assert is_image_too_dark(fixture_dark_image, pixel_brightness_threshold=10.0)


def test_is_image_too_dark_with_bright_image(fixture_bright_image: Image):
    assert not is_image_too_dark(fixture_bright_image, pixel_brightness_threshold=10.0)


def test_is_image_too_dark_with_mid_brightness_image(
    fixture_mid_brightness_image: Image,
):
    assert not is_image_too_dark(
        fixture_mid_brightness_image, pixel_brightness_threshold=10.0
    )


def test_adjust_image_brightness_with_dark_image(fixture_dark_image: Image):
    adjusted_image = adjust_image_brightness(fixture_dark_image)
    # we expect that image to be too dark (it's all dark, so there's no adjustments)
    assert is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_image_brightness_with_bright_image(fixture_bright_image: Image):
    adjusted_image = adjust_image_brightness(fixture_bright_image)
    # Since the image was already bright, it should remain bright
    assert not is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_image_brightness_with_mid_brightness_image(
    fixture_mid_brightness_image: Image,
):
    adjusted_image = adjust_image_brightness(fixture_mid_brightness_image)
    # The image should still not be too dark after adjustment
    assert not is_image_too_dark(adjusted_image, pixel_brightness_threshold=10.0)


def test_adjust_nuclear_speckle_image_brightness(
    fixture_nuclear_speckle_example_image: Image,
):
    assert is_image_too_dark(fixture_nuclear_speckle_example_image)
    assert not is_image_too_dark(
        adjust_image_brightness(fixture_nuclear_speckle_example_image),
        pixel_brightness_threshold=3.0,
    )


@pytest.mark.parametrize(
    "orig_image, outline_image, expected_non_black_mask",
    [
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # All-black original image
            np.full((10, 10, 3), 255, dtype=np.uint8),  # White outline image
            True,  # All pixels are non-black
        ),
        (
            np.zeros((10, 10), dtype=np.uint8),  # Grayscale original image
            np.full((10, 10), 255, dtype=np.uint8),  # Grayscale white outline
            True,  # All pixels are non-black
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB original image
            np.zeros((5, 5, 3), dtype=np.uint8),  # Mismatched size outline
            False,  # All pixels remain black after resizing
        ),
    ],
)
def test_draw_outline_on_image_from_outline(
    tmp_path: pathlib.Path,
    orig_image: np.ndarray,
    outline_image: np.ndarray,
    expected_non_black_mask: bool,
) -> None:
    """
    Tests draw_outline_on_image_from_outline.
    """
    # Save the outline image to a temporary path
    outline_image_path = tmp_path / "outline.png"
    imsave(outline_image_path, outline_image)

    # Call the method
    result_image = draw_outline_on_image_from_outline(
        orig_image, str(outline_image_path)
    )

    # Validate results
    non_black_mask = np.any(result_image[..., :3] != 0, axis=-1)

    if expected_non_black_mask:
        assert np.any(non_black_mask), "Expected a non-black outline but got none."
    else:
        assert not np.any(
            non_black_mask
        ), "Expected no outline but got a non-black area."


@pytest.mark.parametrize(
    "orig_image, mask_image, expected_outlines",
    [
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB black original image
            np.zeros((10, 10), dtype=np.uint8),  # Binary mask with no objects
            False,  # No outline expected
        ),
        (
            np.zeros((10, 10, 3), dtype=np.uint8),  # RGB black original image
            np.pad(np.ones((6, 6), dtype=np.uint8), 2) * 255,  # Square mask
            True,  # Outline expected
        ),
        (
            np.zeros((10, 10), dtype=np.uint8),  # Grayscale original image
            np.zeros((10, 10), dtype=np.uint8),  # Binary mask with no objects
            False,  # No outline expected
        ),
        (
            np.zeros((20, 20, 3), dtype=np.uint8),  # Larger RGB black original image
            np.zeros((20, 20), dtype=np.uint8),  # Binary mask with a circle
            True,  # Outline expected
        ),
    ],
)
def test_draw_outline_on_image_from_mask(
    tmp_path: pathlib.Path,
    orig_image: np.ndarray,
    mask_image: np.ndarray,
    expected_outlines: bool,
) -> None:
    """
    Tests draw_outline_on_image_from_mask.
    """
    # Create a valid circular mask for case 3
    if mask_image.shape == (20, 20) and expected_outlines:
        rr, cc = disk((10, 10), 5)
        mask_image[rr, cc] = 255

    # Save the mask image to a temporary path
    mask_image_path = tmp_path / "mask.png"
    imsave(mask_image_path, mask_image)

    # Call the method
    result_image = draw_outline_on_image_from_mask(orig_image, str(mask_image_path))

    # Check for green outlines in the result
    green_color = [0, 255, 0]
    mask = (
        (result_image == green_color).all(axis=-1) if result_image.ndim == 3 else None
    )

    if expected_outlines:
        assert mask is not None and mask.any(), "Expected outlines but found none."
    else:
        assert mask is None or not mask.any(), "Unexpected outlines found."


@pytest.mark.parametrize(
    "input_image, expected_mode, expected_shape",
    [
        # Grayscale image
        (Image.fromarray(np.zeros((100, 100), dtype=np.uint8)), "L", (100, 100)),
        # RGB image
        (
            Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)),
            "RGB",
            (100, 100, 3),
        ),
        # RGBA image
        (
            Image.fromarray(np.zeros((100, 100, 4), dtype=np.uint8)),
            "RGBA",
            (100, 100, 4),
        ),
    ],
)
def test_adjust_with_adaptive_histogram_equalization(
    input_image: Image.Image, expected_mode: str, expected_shape: Tuple[int, ...]
) -> None:
    """
    Tests adjust_with_adaptive_histogram_equalization.
    """
    # Call the function
    output_image = adjust_with_adaptive_histogram_equalization(input_image)

    # Check that the output is a PIL Image
    assert isinstance(output_image, Image.Image), "Output is not a PIL Image."

    # Check that the mode is as expected
    assert (
        output_image.mode == expected_mode
    ), f"Expected mode {expected_mode}, got {output_image.mode}."

    # Convert output image to NumPy array and check its shape
    output_array = np.asarray(output_image)
    assert (
        output_array.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {output_array.shape}."

    # Check that the output is not identical to the input
    input_array = np.asarray(input_image)
    assert not np.array_equal(
        input_array, output_array
    ), "Output image is identical to input, no adjustment applied."
