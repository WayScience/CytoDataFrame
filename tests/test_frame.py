"""
Tests cosmicqc CytoDataFrame module
"""

import pathlib
from io import BytesIO

import cosmicqc
import numpy as np
import pandas as pd
import plotly
import plotly.colors as pc
import pytest
from pyarrow import parquet

from cytodataframe.frame import CytoDataFrame
from tests.utils import (
    create_sample_image,
    create_sample_outline,
    cytodataframe_image_display_contains_green_pixels,
)


def test_cytodataframe_input(
    tmp_path: pathlib.Path,
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    basic_outlier_csv_gz: str,
    basic_outlier_tsv: str,
    basic_outlier_parquet: str,
):
    # Tests CytoDataFrame with pd.DataFrame input.
    sc_df = CytoDataFrame(data=basic_outlier_dataframe)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == "pandas.DataFrame"
    assert sc_df.equals(basic_outlier_dataframe)

    # test export
    basic_outlier_dataframe.to_parquet(
        control_path := f"{tmp_path}/df_input_example.parquet"
    )
    sc_df.export(test_path := f"{tmp_path}/df_input_example1.parquet")

    assert parquet.read_table(control_path).equals(parquet.read_table(test_path))

    # Tests CytoDataFrame with pd.Series input.
    sc_df = CytoDataFrame(data=basic_outlier_dataframe.loc[0])

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == "pandas.Series"
    assert sc_df.equals(pd.DataFrame(basic_outlier_dataframe.loc[0]))

    # Tests CytoDataFrame with CSV input.
    sc_df = CytoDataFrame(data=basic_outlier_csv)
    expected_df = pd.read_csv(basic_outlier_csv)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_csv)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path))

    # Tests CytoDataFrame with CSV input.
    sc_df = CytoDataFrame(data=basic_outlier_csv_gz)
    expected_df = pd.read_csv(basic_outlier_csv_gz)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_csv_gz)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.csv.gz", index=False)

    pd.testing.assert_frame_equal(
        expected_df, pd.read_csv(test_path, compression="gzip")
    )

    # Tests CytoDataFrame with TSV input.
    sc_df = CytoDataFrame(data=basic_outlier_tsv)
    expected_df = pd.read_csv(basic_outlier_tsv, delimiter="\t")

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_tsv)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example.tsv", index=False)

    pd.testing.assert_frame_equal(expected_df, pd.read_csv(test_path, sep="\t"))

    # Tests CytoDataFrame with parquet input.
    sc_df = CytoDataFrame(data=basic_outlier_parquet)
    expected_df = pd.read_parquet(basic_outlier_parquet)

    # test that we ingested the data properly
    assert sc_df._custom_attrs["data_source"] == str(basic_outlier_parquet)
    assert sc_df.equals(expected_df)

    # test export
    sc_df.export(test_path := f"{tmp_path}/df_input_example2.parquet")

    assert parquet.read_table(basic_outlier_parquet).equals(
        parquet.read_table(test_path)
    )

    # test CytoDataFrame with CytoDataFrame input
    copy_sc_df = CytoDataFrame(data=sc_df)

    pd.testing.assert_frame_equal(copy_sc_df, sc_df)


def test_show_report(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Used for testing show report capabilities
    """

    df = cosmicqc.analyze.label_outliers(
        df=cytotable_CFReT_data_df,
        include_threshold_scores=True,
    )

    figures = df.show_report(auto_open=False)

    expected_number_figures = 3
    assert len(figures) == expected_number_figures
    assert (
        next(iter({type(figure) for figure in figures}))
        == plotly.graph_objs._figure.Figure
    )

    df.show_report(
        report_path=(report_path := pathlib.Path("cosmicqc_example_report.html")),
        auto_open=False,
    )

    assert report_path.is_file()


def test_repr_html(
    cytotable_NF1_data_parquet_shrunken: str,
    cytotable_nuclear_speckles_data_parquet: str,
):
    """
    Tests how images are rendered through customized repr_html in CytoDataFrame.
    """

    # Ensure there's at least one greenish pixel in the image
    assert cytodataframe_image_display_contains_green_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
    ), "The NF1 images do not contain green outlines."
    assert cytodataframe_image_display_contains_green_pixels(
        frame=CytoDataFrame(
            data=cytotable_nuclear_speckles_data_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_nuclear_speckles_data_parquet).parent}/masks",
        ),
        image_cols=[
            "Image_FileName_A647",
            "Image_FileName_DAPI",
            "Image_FileName_GOLD",
        ],
    ), "The nuclear speckles images do not contain green outlines."


@pytest.mark.generate_report_image
def fixture_generate_show_report_html_output(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Used for generating report output for use with other tests.
    """

    # create outliers dataframe
    df = cosmicqc.analyze.label_outliers(
        df=cytotable_CFReT_data_df,
        include_threshold_scores=True,
    )

    # show a report
    df.show_report(
        report_path=(
            report_path := pathlib.Path(__file__).parent
            / "data"
            / "coSMicQC"
            / "show_report"
            / "cosmicqc_example_report.html"
        ),
        color_palette=pc.qualitative.Dark24[0:2],
        auto_open=False,
    )

    return report_path


def test_overlay_with_valid_images():
    """
    Tests the `draw_outline_on_image_from_outline` function
    with valid images: a base image and an outline image.

    Verifies that the resulting image contains the correct
    outline color in the expected positions.
    """
    # Create a sample base image (black background)
    actual_image = create_sample_image(200, 200, (0, 0, 0, 255))  # Black image
    outline_image = create_sample_outline(200, 200, (255, 0, 0))  # Red outline

    # Save images to bytes buffer (to mimic files)
    actual_image_fp = BytesIO()
    actual_image.save(actual_image_fp, format="PNG")
    actual_image_fp.seek(0)

    outline_image_fp = BytesIO()
    outline_image.save(outline_image_fp, format="PNG")
    outline_image_fp.seek(0)

    # Test the function
    result_image = CytoDataFrame.draw_outline_on_image_from_outline(
        actual_image_fp, outline_image_fp
    )

    # Convert result to numpy array for comparison
    result_array = np.array(result_image)

    # Assert that the result image has the outline color
    # (e.g., red) in the expected position
    assert np.any(
        result_array[10:100, 10:100, :3] == [255, 0, 0]
    )  # Check for red outline
    assert np.all(
        result_array[0:10, 0:10, :3] == [0, 0, 0]
    )  # Check for no outline in the black background


def test_overlay_with_no_outline():
    """
    Tests the `draw_outline_on_image_from_outline` function
    with an outline image that has no outlines (all black).

    Verifies that the result is the same as the original
    image when no outlines are provided.
    """
    # Create a sample base image
    actual_image = create_sample_image(200, 200, (0, 0, 255, 255))
    # Black image with no outline
    outline_image = create_sample_image(200, 200, (0, 0, 0, 255))

    actual_image_fp = BytesIO()
    actual_image.save(actual_image_fp, format="PNG")
    actual_image_fp.seek(0)

    outline_image_fp = BytesIO()
    outline_image.save(outline_image_fp, format="PNG")
    outline_image_fp.seek(0)

    # Test the function
    result_image = CytoDataFrame.draw_outline_on_image_from_outline(
        actual_image_fp, outline_image_fp
    )

    # Convert result to numpy array for comparison
    result_array = np.array(result_image)

    # Assert that the result image is still blue (no outline overlay)
    assert np.all(result_array[:, :, :3] == [0, 0, 255])


def test_overlay_with_transparent_outline():
    """
    Tests the `draw_outline_on_image_from_outline` function
    with a fully transparent outline image.

    Verifies that the result image is unchanged when the
    outline image is fully transparent.
    """
    # Create a sample base image
    actual_image = create_sample_image(200, 200, (0, 255, 0, 255))
    # Fully transparent image
    outline_image = create_sample_image(200, 200, (0, 0, 0, 0))

    actual_image_fp = BytesIO()
    actual_image.save(actual_image_fp, format="PNG")
    actual_image_fp.seek(0)

    outline_image_fp = BytesIO()
    outline_image.save(outline_image_fp, format="PNG")
    outline_image_fp.seek(0)

    # Test the function
    result_image = CytoDataFrame.draw_outline_on_image_from_outline(
        actual_image_fp, outline_image_fp
    )

    # Convert result to numpy array for comparison
    result_array = np.array(result_image)

    # Assert that the result image is still green
    # (transparent outline should not affect the image)
    assert np.all(result_array[:, :, :3] == [0, 255, 0])


def test_invalid_image_path():
    """
    Tests the `draw_outline_on_image_from_outline` function
    when the image path is invalid.

    Verifies that a FileNotFoundError is raised when the
    specified image does not exist.
    """
    with pytest.raises(FileNotFoundError):
        CytoDataFrame.draw_outline_on_image_from_outline(
            "invalid_image.png", "valid_outline.png"
        )


def test_invalid_outline_path():
    """
    Tests the `draw_outline_on_image_from_outline` function
    when the outline image path is invalid.

    Verifies that a FileNotFoundError is raised when the
    specified outline image does not exist.
    """
    with pytest.raises(FileNotFoundError):
        CytoDataFrame.draw_outline_on_image_from_outline(
            "valid_image.png", "invalid_outline.png"
        )
