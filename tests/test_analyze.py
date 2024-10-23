"""
Tests cosmicqc analyze module
"""

import pandas as pd
import pytest
from cosmicqc import analyze


def test_find_outliers_basic_dataframe(basic_outlier_dataframe: pd.DataFrame):
    """
    Testing find_outliers with basic/simulated data.
    """

    # add metadata to basic data
    metadata_column_name = "Image_Metadata_Plate"
    basic_outlier_dataframe[metadata_column_name] = "A"

    # assert that we have the output we expect
    assert analyze.find_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        metadata_columns=[metadata_column_name],
    ).to_dict(orient="dict") == {
        "example_feature": {8: 9, 9: 10},
        "Image_Metadata_Plate": {8: "A", 9: "A"},
    }


def test_find_outliers_basic_csv(basic_outlier_csv: str):
    """
    Testing find_outliers with csv data.
    """

    # assert that we have the output we expect
    assert analyze.find_outliers(
        df=basic_outlier_csv,
        feature_thresholds={"example_feature": 1},
        metadata_columns=[],
    ).to_dict(orient="dict") == {
        "example_feature": {8: 9, 9: 10},
    }


def test_find_outliers_cfret(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Testing find_outliers with CytoTable CFReT data.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # Set a negative threshold to identify both outlier small nuclei
    # and low formfactor representing non-circular segmentations.
    feature_thresholds = {
        "Nuclei_AreaShape_Area": -1,
        "Nuclei_AreaShape_FormFactor": -1,
    }

    # run function to identify outliers given conditions
    small_area_formfactor_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we found the appropriate outliers
    assert small_area_formfactor_outliers_df.sort_values(
        list(feature_thresholds)
    ).to_dict(orient="dict") == {
        "Nuclei_AreaShape_Area": {7802: 752.0, 11967: 788.0, 5626: 796.0, 9238: 798.0},
        "Nuclei_AreaShape_FormFactor": {
            7802: 0.7981428715244236,
            11967: 0.5476435143650794,
            5626: 0.7348718757398186,
            9238: 0.8202563209583683,
        },
        "Image_Metadata_Plate": {
            7802: "localhost231120090001",
            11967: "localhost231120090001",
            5626: "localhost231120090001",
            9238: "localhost231120090001",
        },
        "Image_Metadata_Well": {7802: "D05", 11967: "E04", 5626: "C09", 9238: "D07"},
        "Image_Metadata_Site": {7802: "f03", 11967: "f10", 5626: "f13", 9238: "f04"},
    }

    # find very elongated nuclei segmentations (above mean)
    feature_thresholds = {
        "Nuclei_AreaShape_Eccentricity": 2,
    }

    # run function to identify outliers given conditions
    eccent_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we found the appropriate outliers
    assert eccent_outliers_df.sort_values(list(feature_thresholds)).to_dict(
        orient="dict"
    ) == {
        "Nuclei_AreaShape_Eccentricity": {
            7802: 0.8459531594205444,
            7884: 0.8528083158737935,
            5626: 0.8571246429020986,
            20609: 0.871038739898089,
            13920: 0.873104711473235,
            10061: 0.8755073203769763,
            10416: 0.876810524700015,
            4978: 0.8768628647595129,
            19420: 0.876866752777687,
            11967: 0.9109918316434343,
        },
        "Image_Metadata_Plate": {
            7802: "localhost231120090001",
            7884: "localhost231120090001",
            5626: "localhost231120090001",
            20609: "localhost231120090001",
            13920: "localhost231120090001",
            10061: "localhost231120090001",
            10416: "localhost231120090001",
            4978: "localhost231120090001",
            19420: "localhost231120090001",
            11967: "localhost231120090001",
        },
        "Image_Metadata_Well": {
            7802: "D05",
            7884: "D05",
            5626: "C09",
            20609: "G08",
            13920: "E08",
            10061: "D09",
            10416: "D11",
            4978: "C05",
            19420: "G07",
            11967: "E04",
        },
        "Image_Metadata_Site": {
            7802: "f03",
            7884: "f10",
            5626: "f13",
            20609: "f04",
            13920: "f07",
            10061: "f03",
            10416: "f11",
            4978: "f13",
            19420: "f00",
            11967: "f10",
        },
    }

    # find large nuclei segmentations (above mean) and low formfactor
    feature_thresholds = {"Nuclei_AreaShape_Area": 2, "Nuclei_AreaShape_FormFactor": -2}

    # run function to identify outliers given conditions
    large_area_formfactor_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    assert large_area_formfactor_outliers_df.sort_values(
        list(feature_thresholds)
    ).to_dict(orient="dict") == {
        "Nuclei_AreaShape_Area": {
            20729: 1933.0,
            7796: 2098.0,
            14825: 2365.0,
            13920: 2664.0,
            9159: 2708.0,
            14178: 3498.0,
            10066: 3640.0,
            14811: 3751.0,
        },
        "Nuclei_AreaShape_FormFactor": {
            20729: 0.723128135340047,
            7796: 0.659333509302755,
            14825: 0.6794834892343651,
            13920: 0.5464078879768164,
            9159: 0.48231330364709524,
            14178: 0.41227682658167264,
            10066: 0.6545633283748163,
            14811: 0.6262476165070433,
        },
        "Image_Metadata_Plate": {
            20729: "localhost231120090001",
            7796: "localhost231120090001",
            14825: "localhost231120090001",
            13920: "localhost231120090001",
            9159: "localhost231120090001",
            14178: "localhost231120090001",
            10066: "localhost231120090001",
            14811: "localhost231120090001",
        },
        "Image_Metadata_Well": {
            20729: "G10",
            7796: "D05",
            14825: "F02",
            13920: "E08",
            9159: "D07",
            14178: "F02",
            10066: "D09",
            14811: "F02",
        },
        "Image_Metadata_Site": {
            20729: "f04",
            7796: "f03",
            14825: "f01",
            13920: "f07",
            9159: "f02",
            14178: "f03",
            10066: "f03",
            14811: "f01",
        },
    }


def test_read_thresholds_set_from_file():
    """
    Tests read_thresholds_set_from_file
    """

    # test that an exception is raised on receiving a bad
    # lookup value from the thresholds file.
    with pytest.raises(LookupError):
        analyze.read_thresholds_set_from_file(
            feature_thresholds="bad_lookup_value",
            feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
        )

    # test default threshold sets
    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="small_and_low_formfactor_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Area": -1, "Nuclei_AreaShape_FormFactor": -1}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="elongated_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Eccentricity": 2}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="large_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Area": 2, "Nuclei_AreaShape_FormFactor": -2}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {
        "small_and_low_formfactor_nuclei": {
            "Nuclei_AreaShape_Area": -1,
            "Nuclei_AreaShape_FormFactor": -1,
        },
        "elongated_nuclei": {"Nuclei_AreaShape_Eccentricity": 2},
        "large_nuclei": {"Nuclei_AreaShape_Area": 2, "Nuclei_AreaShape_FormFactor": -2},
    }


def test_find_outliers_dict_and_default_config_cfret(
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Testing find_outliers with dictionary vs yaml threshold sets
    using CytoTable CFReT data.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Area": -1,
                "Nuclei_AreaShape_FormFactor": -1,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="small_and_low_formfactor_nuclei",
            metadata_columns=metadata_columns,
        ),
    )

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Eccentricity": 2,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="elongated_nuclei",
            metadata_columns=metadata_columns,
        ),
    )

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Area": 2,
                "Nuclei_AreaShape_FormFactor": -2,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="large_nuclei",
            metadata_columns=metadata_columns,
        ),
    )


def test_label_outliers(
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Tests label_outliers
    """

    # compare the dataframe vs csv output to make sure they are equivalent
    pd.testing.assert_frame_equal(
        analyze.label_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
        analyze.label_outliers(
            df=basic_outlier_csv,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
    )

    # test basic single-column result with zscores
    assert analyze.label_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=True,
    ).to_dict(orient="dict") == {
        "example_feature": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        },
        "cqc.custom.Z_Score.example_feature": {
            0: -1.5666989036012806,
            1: -1.2185435916898848,
            2: -0.8703882797784892,
            3: -0.5222329678670935,
            4: -0.17407765595569785,
            5: 0.17407765595569785,
            6: 0.5222329678670935,
            7: 0.8703882797784892,
            8: 1.2185435916898848,
            9: 1.5666989036012806,
        },
        "cqc.custom.is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    # test for case when zscores are excluded
    assert analyze.label_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=False,
    ).to_dict(orient="dict") == {
        "example_feature": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        },
        "cqc.custom.is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    # test single-column result
    test_df = analyze.label_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds="large_nuclei",
        include_threshold_scores=True,
    )
    pd.testing.assert_frame_equal(
        test_df,
        pd.read_parquet(
            path="tests/data/coSMicQC/output_data/test_label_outliers_output.parquet",
            columns=test_df.columns.tolist(),
        ),
    )

    # test full dataset
    pd.testing.assert_frame_equal(
        analyze.label_outliers(
            df=cytotable_CFReT_data_df, include_threshold_scores=True
        ),
        pd.read_parquet(
            path="tests/data/coSMicQC/output_data/test_label_outliers_output.parquet"
        ),
    )


def test_identify_outliers(
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Tests identify_outliers
    """

    # show that dataframe and csv output are the same
    pd.testing.assert_frame_equal(
        analyze.identify_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
        analyze.identify_outliers(
            df=basic_outlier_csv,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
    )

    assert analyze.identify_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=True,
    ).to_dict(orient="dict") == {
        "cqc.custom.Z_Score.example_feature": {
            0: -1.5666989036012806,
            1: -1.2185435916898848,
            2: -0.8703882797784892,
            3: -0.5222329678670935,
            4: -0.17407765595569785,
            5: 0.17407765595569785,
            6: 0.5222329678670935,
            7: 0.8703882797784892,
            8: 1.2185435916898848,
            9: 1.5666989036012806,
        },
        "cqc.custom.is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    pd.testing.assert_frame_equal(
        analyze.identify_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="large_nuclei",
            include_threshold_scores=True,
        ),
        pd.read_parquet(
            "tests/data/coSMicQC/output_data/test_identifier_outliers_output.parquet"
        ),
    )

    identified_df = analyze.identify_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds="large_nuclei",
    )
    pd.testing.assert_series_equal(
        identified_df,
        pd.read_parquet(
            "tests/data/coSMicQC/output_data/test_identifier_outliers_output.parquet",
            columns=["cqc.large_nuclei.is_outlier"],
        )["cqc.large_nuclei.is_outlier"],
        check_names=False,
    )
