"""
Tests project integration with other packages
"""

import pandas as pd
import cosmicqc


def test_find_outliers_cfret(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Testing cosmicqc.find_outliers with CytoTable CFReT data.
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
    small_area_formfactor_outliers_df = cosmicqc.analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we found the appropriate outliers
    assert (
        type(
            small_area_formfactor_outliers_df.sort_values(
                list(feature_thresholds)
            ).to_dict(orient="dict")
        )
        == dict
    )
