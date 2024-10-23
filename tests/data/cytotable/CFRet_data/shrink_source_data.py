"""
Module to shrink source data for testing.

Original source of data:
https://github.com/WayScience/CFReT_data/blob/
main/3.process_cfret_features/data/
converted_profiles/localhost231120090001_converted.parquet
"""

import os

import pandas as pd

# note: we assume the dataset has been manually added to the
# directory containing this module.
filename = f"{os.path.dirname(__file__)}/localhost231120090001_converted.parquet"

# read the data from parquet, sample a fraction of the data
df = pd.read_parquet(filename).sample(frac=0.03, replace=True, random_state=1)

# export to a new file
df.to_parquet(
    f"{os.path.dirname(__file__)}/test_localhost231120090001_converted.parquet"
)
