"""
Module to shrink source data for testing.

Original source of data (processing):
https://github.com/WayScience/JUMP-single-cell
"""

import duckdb
from pyarrow import parquet

# gather data from cytotable output which includes an example image
with duckdb.connect() as ddb:
    result = ddb.execute(
        """
        SELECT * FROM read_parquet('BR00117006.parquet')
        WHERE Image_FileName_OrigAGP = 'r01c01f01p01-ch2sk1fk1fl1.tiff'
        LIMIT 5;
        """
    ).arrow()

# write the output to a shrunken parquet file
parquet.write_table(table=result, where="BR00117006_shrunken.parquet")
