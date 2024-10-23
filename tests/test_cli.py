"""
Tests cosmicqc cli module
"""

import pathlib

from pyarrow import parquet

from .utils import run_cli_command


def test_cli_util():
    """
    Test the run_cli_command for successful output
    """

    command = """echo 'hello world'"""
    stdout, stderr, returncode = run_cli_command(command)

    assert returncode == 0


def test_cli_identify_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `identify_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc identify_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/identify_outliers_output.parquet"
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """0    False
1    False
2    False
3    False
4    False
5    False
6    False
7    False
8     True
9     True
Name: cqc.custom.Z_Score.example_feature, dtype: bool""".strip()
    )

    assert parquet.read_table(
        f"{tmp_path}/identify_outliers_output.parquet"
    ).to_pydict() == {
        "cqc.custom.Z_Score.example_feature": [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ]
    }


def test_cli_find_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `find_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc find_outliers --df {basic_outlier_csv}"""
            """ --metadata_columns [] --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/find_outliers_output.parquet"
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """Number of outliers: 2 (20.00%)
Outliers Range:
example_feature Min: 9
example_feature Max: 10
   example_feature
8                9
9               10""".strip()
    )

    assert parquet.read_table(
        f"{tmp_path}/find_outliers_output.parquet"
    ).to_pydict() == {"example_feature": [9, 10], "__index_level_0__": [8, 9]}


def test_cli_label_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `label_outliers` function of the CLI.
    """

    stdout, stderr, returncode = run_cli_command(
        (
            f"""cosmicqc label_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/label_outliers_output.parquet"
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """example_feature  cqc.custom.is_outlier
0                1                  False
1                2                  False
2                3                  False
3                4                  False
4                5                  False
5                6                  False
6                7                  False
7                8                  False
8                9                   True
9               10                   True""".strip()
    )

    assert parquet.read_table(
        f"{tmp_path}/label_outliers_output.parquet"
    ).to_pydict() == {
        "example_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "cqc.custom.is_outlier": [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    }
