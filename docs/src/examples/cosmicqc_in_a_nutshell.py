# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # `coSMicQC` in a nutshell
#
# This notebook demonstrates various capabilities of `coSMicQC` using examples.

# +
import pathlib

import cosmicqc
import pandas as pd

# set a path for the parquet-based dataset
# (in this case, CellProfiler SQLite data processed by CytoTable)
data_path = (
    "../../../tests/data/cytotable/NF1_cellpainting_data/"
    "Plate_2_with_image_data.parquet"
)

# set a context directory for images associated with the dataset
image_context_dir = pathlib.Path(data_path).parent / "Plate_2_images"
mask_context_dir = pathlib.Path(data_path).parent / "Plate_2_masks"

# create a cosmicqc CytoDataFrame (single-cell DataFrame)
scdf = cosmicqc.CytoDataFrame(
    data=data_path,
    data_context_dir=image_context_dir,
    data_mask_context_dir=mask_context_dir,
)

# display the dataframe
scdf
# -

# Identify which rows include outliers for a given threshold definition
# which references a column name and a z-score number which is considered
# the limit.
cosmicqc.analyze.identify_outliers(
    df=scdf,
    feature_thresholds={"Nuclei_AreaShape_Area": -1},
).sort_values()

# Show the number of outliers given a column name and a specified threshold
# via the `feature_thresholds` parameter and the `find_outliers` function.
cosmicqc.analyze.find_outliers(
    df=scdf,
    metadata_columns=["Metadata_ImageNumber", "Image_Metadata_Plate_x"],
    feature_thresholds={"Nuclei_AreaShape_Area": -1},
)

# +
# create a labeled dataset which includes z-scores and whether those scores
# are interpreted as outliers or inliers. We use pre-defined threshold sets
# loaded from defaults (cosmicqc can accept user-defined thresholds too!).
labeled_scdf = cosmicqc.analyze.label_outliers(
    df=scdf,
    include_threshold_scores=True,
)

# show the dataframe rows with only the last 8 columns
# (added from the label_outliers function)
labeled_scdf.iloc[:, -8:]
# -

# show histogram reports on the outliers and inliers
# for each threshold set in the new columns
labeled_scdf.show_report()  # fmt: skip

# show cropped images through CytoDataFrame from the dataset to help analyze outliers
labeled_scdf.sort_values(by="cqc.large_nuclei.is_outlier", ascending=False)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "cqc.large_nuclei.is_outlier",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
]

# One can convert from cosmicqc.CytoDataFrame to pd.DataFrame's
# (when or if needed!)
df = pd.DataFrame(scdf)
print(type(df))
df
