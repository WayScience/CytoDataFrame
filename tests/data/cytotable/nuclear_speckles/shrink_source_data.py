"""
Module to shrink source data for testing.

Original source of data (processing):
https://github.com/WayScience/nuclear_speckle_image_profiling
"""

import os

import pandas as pd

# note: we assume the dataset has been manually added to the
# directory containing this module.
filename = f"{os.path.dirname(__file__)}/slide1_converted.parquet"

# read the data from parquet, sample a fraction of the data
df = pd.read_parquet(filename)

# filter to only those data which include slide1_A1_M10_CH0_Z09_illumcorrect
df = df[
    (
        df["Image_FileName_A647"].str.contains(
            img_str := "slide1_A1_M10_CH0_Z09_illumcorrect"
        )
    )
    | (df["Image_FileName_DAPI"].str.contains(img_str))
    | (df["Image_FileName_GOLD"].str.contains(img_str))
]

# export to a new file
df.to_parquet(f"{os.path.dirname(__file__)}/test_slide1_converted.parquet")
