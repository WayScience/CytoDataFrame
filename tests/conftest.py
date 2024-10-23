"""
Fixtures for testing via pytest.
See here for more information:
https://docs.pytest.org/en/7.1.x/explanation/fixtures.html
"""

import pathlib

import cosmicqc
import numpy as np
import pandas as pd
import plotly.colors as pc
import pytest
import skimage
from PIL import Image


@pytest.fixture(name="cytotable_CFReT_data_df")
def fixture_cytotable_CFReT_df():
    """
    Return df to test CytoTable CFReT_data
    """
    return pd.read_parquet(
        "tests/data/cytotable/CFRet_data/test_localhost231120090001_converted.parquet"
    )


@pytest.fixture(name="cytotable_NF1_data_parquet_shrunken")
def fixture_cytotable_NF1_data_parquet_shrunken():
    """
    Return df to test CytoTable NF1 data through shrunken parquet file
    """
    return (
        "tests/data/cytotable/NF1_cellpainting_data_shrunken/"
        "Plate_2_with_image_data_shrunken.parquet"
    )


@pytest.fixture(name="cytotable_nuclear_speckles_data_parquet")
def fixture_cytotable_nuclear_speckle_data_parquet():
    """
    Return df to test CytoTable nuclear speckles data through shrunken parquet file
    """
    return "tests/data/cytotable/nuclear_speckles/test_slide1_converted.parquet"


@pytest.fixture(name="basic_outlier_dataframe")
def fixture_basic_outlier_dataframe():
    """
    Creates basic example data for use in tests
    """
    return pd.DataFrame({"example_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture(name="basic_outlier_csv")
def fixture_basic_outlier_csv(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data csv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        csv_path := tmp_path / "basic_example.csv", index=False
    )

    return csv_path


@pytest.fixture(name="basic_outlier_csv_gz")
def fixture_basic_outlier_csv_gz(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data csv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        csv_gz_path := tmp_path / "example.csv.gz", index=False, compression="gzip"
    )

    return csv_gz_path


@pytest.fixture(name="basic_outlier_tsv")
def fixture_basic_outlier_tsv(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data tsv for use in tests
    """

    basic_outlier_dataframe.to_csv(
        tsv_path := tmp_path / "example.tsv", sep="\t", index=False
    )

    return tsv_path


@pytest.fixture(name="basic_outlier_parquet")
def fixture_basic_outlier_parquet(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Creates basic example data parquet for use in tests
    """

    basic_outlier_dataframe.to_parquet(
        parquet_path := tmp_path / "example.parquet", index=False
    )

    return parquet_path


@pytest.fixture(name="generate_show_report_html_output")
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


@pytest.fixture
def fixture_dark_image():
    # Create a dark image (50x50 pixels, almost black)
    dark_img_array = np.zeros((50, 50, 3), dtype=np.uint8)
    return Image.fromarray(dark_img_array)


@pytest.fixture
def fixture_mid_brightness_image():
    # Create an image with medium brightness (50x50 pixels, mid gray)
    mid_brightness_img_array = np.full((50, 50, 3), 128, dtype=np.uint8)
    return Image.fromarray(mid_brightness_img_array)


@pytest.fixture
def fixture_bright_image():
    # Create a bright image (50x50 pixels, almost white)
    bright_img_array = np.full((50, 50, 3), 255, dtype=np.uint8)
    return Image.fromarray(bright_img_array)


@pytest.fixture
def fixture_nuclear_speckle_example_image():
    # create an image array from example nuclear speckle data
    return Image.fromarray(
        (
            skimage.io.imread(
                "tests/data/cytotable/nuclear_speckles/images/plate1/slide1_A1_M10_CH0_Z09_illumcorrect.tiff"
            )
            / 256
        ).astype(np.uint8)
    ).convert("RGBA")
