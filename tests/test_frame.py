"""
Tests cosmicqc CytoDataFrame module
"""

import pathlib

import pandas as pd
from pyarrow import parquet

from cytodataframe.frame import CytoDataFrame
from tests.utils import (
    cytodataframe_image_display_contains_pixels,
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


def test_repr_html_green_pixels(
    cytotable_NF1_data_parquet_shrunken: str,
    cytotable_nuclear_speckles_data_parquet: str,
    cytotable_pediatric_cancer_atlas_parquet: str,
):
    """
    Tests how images are rendered through customized repr_html in CytoDataFrame.
    """

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the NF1 dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The NF1 images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the NF1 dataset.
    nf1_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_NF1_data_parquet_shrunken
    )
    nf1_dataset_with_modified_image_paths.loc[
        :, ["Image_PathName_DAPI", "Image_PathName_GFP", "Image_PathName_RFP"]
    ] = f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=nf1_dataset_with_modified_image_paths,
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The NF1 images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the nuclear speckles dataset.
    assert cytodataframe_image_display_contains_pixels(
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
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The nuclear speckles images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are set for the pediatric cancer dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_pediatric_cancer_atlas_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig",
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain green outlines."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the pediatric cancer dataset.
    # (tests the regex associations with default image paths)
    pediatric_cancer_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_pediatric_cancer_atlas_parquet
    )
    # fmt: off
    pediatric_cancer_dataset_with_modified_image_paths = (
        pediatric_cancer_dataset_with_modified_image_paths.assign(
        Image_PathName_OrigAGP=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
        Image_PathName_OrigDNA=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
    )
    )
    # fmt: on

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=pediatric_cancer_dataset_with_modified_image_paths,
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": 255, "red": None, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain green outlines."


def test_repr_html_red_pixels(
    cytotable_NF1_data_parquet_shrunken: str,
    cytotable_nuclear_speckles_data_parquet: str,
    cytotable_pediatric_cancer_atlas_parquet: str,
):
    """
    Tests how images are rendered through customized repr_html in CytoDataFrame.
    """

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the NF1 dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images do not contain red dots."

    # Ensure there are no reddish pixels in the image
    # when context dirs are set for the NF1 dataset.
    assert not cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_NF1_data_parquet_shrunken,
            data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
            compartment_center_xy=False,
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images contain red pixels when it shouldn't."

    # Ensure there's at least one greenish pixel in the image
    # when context dirs are NOT set for the NF1 dataset.
    nf1_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_NF1_data_parquet_shrunken
    )
    nf1_dataset_with_modified_image_paths.loc[
        :, ["Image_PathName_DAPI", "Image_PathName_GFP", "Image_PathName_RFP"]
    ] = f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images"

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=nf1_dataset_with_modified_image_paths,
            data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        ),
        image_cols=["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The NF1 images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the nuclear speckles dataset.
    assert cytodataframe_image_display_contains_pixels(
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
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The nuclear speckles images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are set for the pediatric cancer dataset.
    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=cytotable_pediatric_cancer_atlas_parquet,
            data_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig",
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain red dots."

    # Ensure there's at least one reddish pixel in the image
    # when context dirs are NOT set for the pediatric cancer dataset.
    # (tests the regex associations with default image paths)
    pediatric_cancer_dataset_with_modified_image_paths = pd.read_parquet(
        path=cytotable_pediatric_cancer_atlas_parquet
    )
    # fmt: off
    pediatric_cancer_dataset_with_modified_image_paths = (
        pediatric_cancer_dataset_with_modified_image_paths.assign(
        Image_PathName_OrigAGP=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
        Image_PathName_OrigDNA=(
            f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/orig"
        ),
    )
    )
    # fmt: on

    assert cytodataframe_image_display_contains_pixels(
        frame=CytoDataFrame(
            data=pediatric_cancer_dataset_with_modified_image_paths,
            data_outline_context_dir=f"{pathlib.Path(cytotable_pediatric_cancer_atlas_parquet).parent}/images/outlines",
            segmentation_file_regex={
                r"CellsOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch3.*\.tiff",
                r"NucleiOutlines_BR(\d+)_C(\d{2})_\d+\.tiff": r".*ch5.*\.tiff",
            },
        ),
        image_cols=[
            "Image_FileName_OrigAGP",
            "Image_FileName_OrigDNA",
        ],
        color_conditions={"green": None, "red": 255, "blue": None},
    ), "The pediatric cancer atlas speckles images do not contain red dots."


def test_return_cytodataframe(cytotable_NF1_data_parquet_shrunken: str):
    """
    Tests to ensure we return a CytoDataFrame
    from extended Pandas methods.
    """

    cdf = CytoDataFrame(data=cytotable_NF1_data_parquet_shrunken)

    assert isinstance(cdf.head(), CytoDataFrame)
    assert isinstance(cdf.tail(), CytoDataFrame)
    assert isinstance(cdf.sort_values(by="Metadata_ImageNumber"), CytoDataFrame)
    assert isinstance(cdf.sample(n=5), CytoDataFrame)


def test_cytodataframe_dynamic_width_and_height(
    cytotable_NF1_data_parquet_shrunken: str,
):
    """
    Tests to ensure we return a CytoDataFrame
    from extended Pandas methods.
    """

    cdf = CytoDataFrame(
        data=cytotable_NF1_data_parquet_shrunken,
        data_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_images",
        data_mask_context_dir=f"{pathlib.Path(cytotable_NF1_data_parquet_shrunken).parent}/Plate_2_masks",
        # set the width to 100px and height to auto for images
        display_options={"width": "100px", "height": "auto"},
    )

    # gather the html of the output for the dataframe
    cdf_image_html = cdf[
        ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]
    ][1:2]._repr_html_()

    # test that the html string contains the customized width and height
    # constraints on the 3 images which display within the html output.
    assert cdf_image_html.count("width:100px") == 3
    assert cdf_image_html.count("height:auto") == 3

    # transpose and test for the same to ensure the images are
    # formatted despite being transposed (that we didn't lose them
    # in the process).
    cdf_image_html = cdf[
        ["Image_FileName_DAPI", "Image_FileName_GFP", "Image_FileName_RFP"]
    ][1:2].T._repr_html_()

    assert cdf_image_html.count("width:100px") == 3
    assert cdf_image_html.count("height:auto") == 3
