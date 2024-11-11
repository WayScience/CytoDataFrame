"""
Module to shrink source data for testing.

Original source of data (processing):
https://github.com/WayScience/pediatric_cancer_atlas_profiling
"""

from pathlib import Path

import duckdb
from pyarrow import parquet

# Get the location of the current module using pathlib
module_location = Path(__file__).resolve().parent

# Read the Parquet file's schema
parquet_file = parquet.ParquetFile(f"{module_location}/BR00143976_converted.parquet")

# Get the schema (column names and types)
schema = parquet_file.schema

# Extract the column names
columns = schema.names

# gather data from cytotable output which includes an example image
with duckdb.connect() as ddb:
    result = ddb.execute(
        f"""
        SELECT * FROM read_parquet('{module_location}/BR00143976_converted.parquet')
        WHERE Image_FileName_OrigDNA = 'r03c03f03p01-ch5sk1fk1fl1.tiff'
        LIMIT 5;
        """
    ).arrow()

# write the output to a shrunken parquet file
parquet.write_table(table=result, where="BR00143976_shrunken.parquet")
