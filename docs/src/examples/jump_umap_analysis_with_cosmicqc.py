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

# # JUMP UMAP analysis with coSMicQC
#
# This notebook analyzes [JUMP](https://jump-cellpainting.broadinstitute.org/) data (`cpg0000-jump-pilot`) by leveraging [UMAP](https://arxiv.org/abs/1802.03426) and [coSMicQC](https://github.com/WayScience/coSMicQC).
#
# ## Outline
#
# We focus on a single file from the JUMP dataset: [`BR00117012.sqlite`](https://open.quiltdata.com/b/cellpainting-gallery/tree/cpg0000-jump-pilot/source_4/workspace/backend/2020_11_04_CPJUMP1/BR00117012/BR00117012.sqlite).
# This file is downloaded and prepared by [CytoTable](https://github.com/cytomining/CytoTable) to form a single-cell [Parquet](https://parquet.apache.org/) file which includes all compartment feature data at the single-cell level.
# We use coSMicQC to find and remove erroneous outlier data in order to prepare for UMAP analysis.
# Afterwards, we use UMAP to demonstrate patterns within the data.

# + editable=true slideshow={"slide_type": ""}
import logging
import pathlib
from typing import List, Optional

import cosmicqc
import holoviews
import hvplot.pandas
import numpy as np
import pandas as pd
import plotly.express as px
import pycytominer
import umap
from cytotable.convert import convert
from IPython.display import HTML, Image
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from pyarrow import parquet

# set bokeh for visualizations with hvplot
hvplot.extension("bokeh")

# create a dir for images
pathlib.Path("./images").mkdir(exist_ok=True)

# avoid displaying plot export warnings
logging.getLogger("bokeh.io.export").setLevel(logging.ERROR)

# set the plate name for use throughout the notebook
example_plate = "BR00117012"


# -

# ## Define utility functions for use within this notebook


def generate_umap_embeddings(
    df_input: pd.DataFrame,
    cols_metadata_to_exclude: List[str],
    umap_n_components: int = 2,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Generates UMAP (Uniform Manifold Approximation and Projection)
    embeddings for a given input dataframe,
    excluding specified metadata columns.

    Args:
        df_input (pd.DataFrame]):
            A dataframe which is expected to contain
            numeric columns to be used for UMAP fitting.
        cols_metadata_to_exclude (List[str]):
            A list of column names representing
            metadata columns that should be excluded
            from the UMAP transformation.
        umap_n_components: (int):
            Number of components to use for UMAP.
            Default = 2.
        random_state (int):
            Number to use for random state and
            optional determinism.
            Default = None (random each time)
            Note: values besides None will turn
            off parallelism for umap-learn, likely
            meaning increased processing time.

    Returns:
        np.ndarray:
            A dataframe containing the UMAP embeddings
            with 2 components for each row in the input.
    """

    # Make sure to reinitialize UMAP instance per plate
    umap_fit = umap.UMAP(
        n_components=umap_n_components,
        random_state=random_state,
        # set the default value if we didn't set a random_state
        # otherwise set to 1 (umap-learn will override anyways).
        # this is set to avoid warnings from umap-learn during
        # processing.
        n_jobs=-1 if random_state is None else 1,
    )

    # Fit UMAP and convert to pandas DataFrame
    embeddings = umap_fit.fit_transform(
        X=df_input[
            [
                col
                for col in df_input.columns.tolist()
                if col not in cols_metadata_to_exclude and "cqc." not in col
            ]
            # select only numeric types from the dataframe
        ].select_dtypes(include=[np.number])
    )

    return embeddings


def plot_hvplot_scatter(
    embeddings: np.ndarray,
    title: str,
    filename: str,
    color_dataframe: Optional[pd.DataFrame] = None,
    color_column: Optional[str] = None,
    bgcolor: str = "black",
    cmap: str = "plasma",
    clabel: Optional[str] = None,
) -> holoviews.core.spaces.DynamicMap:
    """
    Creates an outlier-focused scatter hvplot for viewing
    UMAP embedding data with cosmicqc outliers coloration.

    Args:
        embeddings (np.ndarray]):
            A numpy ndarray which includes
            embedding data to display.
        title (str):
            Title for the UMAP scatter plot.
        filename (str):
            Filename which indicates where to export the
            plot.
        color_dataframe (pd.DataFrame):
            A dataframe which includes data used for
            color mapping within the plot. For example,
            coSMicQC .is_outlier columns.
        color_column (str):
            Column name from color_dataframe to use
            for coloring the scatter plot.
        bgcolor (str):
            Sets the background color of the plot.
        cmap (str):
            Sets the colormap used for the plot.
            See here for more:
            https://holoviews.org/user_guide/Colormaps.html
        clabel (str):
            Sets a label on the color map key displayed
            horizontally. Defaults to None (no label).

    Returns:
        holoviews.core.spaces.DynamicMap:
            A dynamic holoviews scatter plot which may be
            displayed in a Jupyter notebook.
    """

    # build a scatter plot through hvplot
    plot = pd.DataFrame(embeddings).hvplot.scatter(
        title=title,
        x="0",
        y="1",
        alpha=0.1,
        rasterize=True,
        c=(
            color_dataframe[color_column].astype(int).values
            if color_dataframe is not None
            else None
        ),
        cnorm="linear",
        cmap=cmap,
        bgcolor=bgcolor,
        height=700,
        width=800,
        clabel=clabel,
    )

    # export the plot
    hvplot.save(obj=plot, filename=filename, center=False)

    return plot


# ## Merge single-cell compartment data into one table

# +
# check if we already have prepared data
if not pathlib.Path(f"./{example_plate}.parquet").is_file():
    # process BR00117012.sqlite using CytoTable to prepare data
    merged_single_cells = convert(
        source_path=(
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace"
            "/backend/2020_11_04_CPJUMP1/BR00117012/BR00117012.sqlite"
        ),
        dest_path=f"./{example_plate}.parquet",
        dest_datatype="parquet",
        source_datatype="sqlite",
        chunk_size=8000,
        preset="cellprofiler_sqlite_cpg0016_jump",
        # allows AWS S3 requests without login
        no_sign_request=True,
        # use explicit cache to avoid temp cache removal
        local_cache_dir="./sqlite_s3_cache/",
        parsl_config=Config(
            executors=[ThreadPoolExecutor(label="tpe_for_jump_processing")]
        ),
        sort_output=False,
    )
else:
    merged_single_cells = f"./{example_plate}.parquet"

# read only the metadata from parquet file
parquet.ParquetFile(merged_single_cells).metadata
# -

# ## Process merged single-cell data using coSMicQC

# + editable=true slideshow={"slide_type": ""}
# show the first few columns for metadata column names
schema_names = parquet.read_schema(merged_single_cells).names
schema_names[:12]

# +
# set a list of metadata columns for use throughout
metadata_cols = [
    "Metadata_ImageNumber",
    "Image_Metadata_Row",
    "Image_Metadata_Site",
    "Metadata_ObjectNumber",
    "Metadata_Plate",
    "Metadata_Well",
    "Image_TableNumber",
]

# read only metadata columns with feature columns used for outlier detection
df_merged_single_cells = pd.read_parquet(
    path=merged_single_cells,
    columns=[
        *metadata_cols,
        "Nuclei_AreaShape_Area",
        "Nuclei_AreaShape_FormFactor",
        "Nuclei_AreaShape_Eccentricity",
    ],
)
df_merged_single_cells.head()
# -

# label outliers within the dataset
print("Large nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="large_nuclei",
)

# label outliers within the dataset
print("Elongated nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="elongated_nuclei",
)

# label outliers within the dataset
print("Small and low formfactor nuclei outliers:")
df_labeled_outliers = cosmicqc.analyze.find_outliers(
    df=df_merged_single_cells,
    metadata_columns=metadata_cols,
    feature_thresholds="small_and_low_formfactor_nuclei",
)

# label outliers within the dataset
df_labeled_outliers = cosmicqc.analyze.label_outliers(
    df=df_merged_single_cells,
    include_threshold_scores=True,
)
# show added columns
df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if "cqc." in col]
].head()

# +
# create a column which indicates whether an erroneous outlier was detected
# from all cosmicqc outlier threshold sets. For ex. True for is_outlier in
# one threshold set out of three would show True for this column. False for
# is_outlier in all threshold sets would show False for this column.
df_labeled_outliers["analysis.included_at_least_one_outlier"] = df_labeled_outliers[
    [col for col in df_labeled_outliers.columns.tolist() if ".is_outlier" in col]
].any(axis=1)

# show value counts for all outliers
outliers_counts = df_labeled_outliers[
    "analysis.included_at_least_one_outlier"
].value_counts()
outliers_counts
# -

# show the percentage of total dataset
print(
    (outliers_counts.iloc[1] / outliers_counts.iloc[0]) * 100,
    "%",
    "of",
    outliers_counts.iloc[0],
    "include erroneous outliers of some kind.",
)

# show histograms to help visualize the data
df_labeled_outliers.show_report();  # fmt: skip

# ## Prepare data for analysis with pycytominer

# +
parquet_sampled_with_outliers = f"./{example_plate}_sampled_with_outliers.parquet"

# check if we already have normalized data
if not pathlib.Path(parquet_sampled_with_outliers).is_file():
    # set a fraction for sampling to limit the amount
    # of data processed based on system memory constraints.
    # note: data was processed on system with 16 CPU, 64 GB ram
    sample_fraction = 0.44

    # read the dataset
    df_features = pd.read_parquet(f"./{example_plate}.parquet")

    # group by metadata_well for all features then sample
    # the dataset by a fraction.
    df_features = (
        # note: we add the column selection here to avoid a pandas
        # DeprecationWarning. See the following link for more details:
        # https://stackoverflow.com/questions/77969964/deprecation-warning-with-groupby-apply
        df_features.groupby(["Metadata_Well"])[df_features.columns]
        .apply(lambda x: x.sample(frac=sample_fraction))
        .reset_index(drop=True)
    )

    # join the sampled feature data with the cosmicqc outlier data
    df_features_with_cqc_outlier_data = df_features.merge(
        # select metadata columns plus those which don't exist in
        # df_features (cosmicqc or analysis-specific columns)
        df_labeled_outliers[
            [
                *metadata_cols,
                *[
                    col
                    for col in df_labeled_outliers.columns
                    if col not in df_features.columns
                ],
            ]
        ],
        how="inner",
        left_on=metadata_cols,
        right_on=metadata_cols,
    )

    df_features_with_cqc_outlier_data.to_parquet(parquet_sampled_with_outliers)

else:
    df_features_with_cqc_outlier_data = pd.read_parquet(parquet_sampled_with_outliers)

df_features_with_cqc_outlier_data
# -

# show our data value counts regarding outliers vs inliers
df_features_with_cqc_outlier_data[
    "analysis.included_at_least_one_outlier"
].value_counts()

# prepare data for normalization and feature selection
# by removing cosmicqc and analaysis focused columns.
df_for_normalize_and_feature_select = df_features_with_cqc_outlier_data[
    # read feature names from cytotable output, which excludes
    # cosmicqc-added columns.
    parquet.read_schema(merged_single_cells).names
]
# show the modified column count
len(df_for_normalize_and_feature_select.columns)

# join JUMP metadata with platemap data to prepare for annotation
df_platemap_and_metadata = pd.read_csv(
    filepath_or_buffer=(
        "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
        "/workspace/metadata/platemaps/2020_11_04_CPJUMP1/"
        "platemap/JUMP-Target-1_compound_platemap.txt"
    ),
    sep="\t",
).merge(
    right=pd.read_csv(
        filepath_or_buffer=(
            "s3://cellpainting-gallery/cpg0000-jump-pilot/source_4"
            "/workspace/metadata/external_metadata/"
            "JUMP-Target-1_compound_metadata.tsv"
        ),
        sep="\t",
    ),
    left_on="broad_sample",
    right_on="broad_sample",
)

# +
parquet_pycytominer_annotated = f"./{example_plate}_annotated.parquet"

# check if we already have annotated data
if not pathlib.Path(parquet_pycytominer_annotated).is_file():
    # annotate the data using pycytominer
    pycytominer.annotate(
        profiles=df_for_normalize_and_feature_select,
        # read the platemap directly from AWS S3 related location
        platemap=df_platemap_and_metadata,
        join_on=["Metadata_well_position", "Metadata_Well"],
        output_file=parquet_pycytominer_annotated,
        output_type="parquet",
    )

# +
parquet_pycytominer_normalized = f"./{example_plate}_normalized.parquet"

# check if we already have normalized data
if not pathlib.Path(parquet_pycytominer_normalized).is_file():
    # normalize the data using pcytominer
    df_pycytominer_normalized = pycytominer.normalize(
        profiles=parquet_pycytominer_annotated,
        features="infer",
        image_features=False,
        meta_features="infer",
        method="standardize",
        samples="Metadata_control_type == 'negcon'",
        output_file=parquet_pycytominer_normalized,
        output_type="parquet",
    )

# +
parquet_pycytominer_feature_selected = f"./{example_plate}_feature_select.parquet"

# check if we already have feature selected data
if not pathlib.Path(parquet_pycytominer_feature_selected).is_file():
    # feature select normalized data using pycytominer
    df_pycytominer_feature_selected = pycytominer.feature_select(
        profiles=parquet_pycytominer_normalized,
        operation=[
            "variance_threshold",
            "correlation_threshold",
            "blocklist",
            "drop_na_columns",
        ],
        na_cutoff=0,
        output_file=parquet_pycytominer_feature_selected,
        output_type="parquet",
    )
# -

# regather metadata columns to account for new additions
all_metadata_cols = [
    col
    for col in parquet.read_schema(parquet_pycytominer_feature_selected).names
    if col.startswith("Metadata_")
]
all_metadata_cols

# calculate UMAP embeddings from the data
# which was prepared by pycytominer.
embeddings_with_outliers = generate_umap_embeddings(
    df_input=pd.read_parquet(parquet_pycytominer_feature_selected),
    cols_metadata_to_exclude=all_metadata_cols,
    random_state=0,
)
# show the shape and top values from the embeddings array
print(embeddings_with_outliers.shape)
embeddings_with_outliers[:3]

plot_hvplot_scatter(
    embeddings=embeddings_with_outliers,
    title=f"UMAP of JUMP embeddings from {example_plate} (with erroneous outliers)",
    filename=(
        image_with_all_outliers
        := f"./images/umap_with_all_outliers_{example_plate}.png"
    ),
    bgcolor="white",
    cmap=px.colors.sequential.Greens[4:],
    clabel="density of single cells",
)
# conserve filespace by displaying export instead of dynamic plot
Image(image_with_all_outliers)

# show a UMAP for all outliers within the data
plot_hvplot_scatter(
    embeddings=embeddings_with_outliers,
    title=f"UMAP of JUMP all coSMicQC erroneous outliers within {example_plate}",
    filename=f"./images/umap_erroneous_outliers_{example_plate}.png",
    color_dataframe=df_features_with_cqc_outlier_data,
    color_column="analysis.included_at_least_one_outlier",
    clabel="density of single cells classified as outliers",
)

# show small and low formfactor nuclei outliers within the data
plot_hvplot_scatter(
    embeddings=embeddings_with_outliers,
    title=f"UMAP of JUMP small and low formfactor nuclei outliers within {example_plate}",
    filename=(
        plot_image
        := f"./images/umap_small_and_low_formfactor_nuclei_outliers_{example_plate}.png"
    ),
    color_dataframe=df_features_with_cqc_outlier_data,
    color_column="cqc.small_and_low_formfactor_nuclei.is_outlier",
    clabel="density of single cells classified as outliers",
)
# conserve filespace by displaying export instead of dynamic plot
Image(plot_image)

# show elongated nuclei outliers within the data
plot_hvplot_scatter(
    embeddings=embeddings_with_outliers,
    title=f"UMAP of JUMP elongated nuclei outliers within {example_plate}",
    filename=(
        plot_image := f"./images/umap_elongated_nuclei_outliers_{example_plate}.png"
    ),
    color_dataframe=df_features_with_cqc_outlier_data,
    color_column="cqc.elongated_nuclei.is_outlier",
    clabel="density of single cells classified as outliers",
)
# conserve filespace by displaying export instead of dynamic plot
Image(plot_image)

# show small and large nuclei outliers within the data
plot_hvplot_scatter(
    embeddings=embeddings_with_outliers,
    title=f"UMAP of JUMP large nuclei outliers within {example_plate}",
    filename=(plot_image := f"./images/umap_large_nuclei_outliers_{example_plate}.png"),
    color_dataframe=df_features_with_cqc_outlier_data,
    color_column="cqc.large_nuclei.is_outlier",
    clabel="density of single cells classified as outliers",
)
# conserve filespace by displaying export instead of dynamic plot
Image(plot_image)

# +
# prepare data for normalization and feature selection
# by removing cosmicqc and analaysis focused columns.
df_for_normalize_and_feature_select_without_outliers = (
    df_features_with_cqc_outlier_data[
        # seek values which are false (not considered an outlier)
        ~df_features_with_cqc_outlier_data["analysis.included_at_least_one_outlier"]
    ][
        # read feature names from cytotable output, which excludes
        # cosmicqc-added columns.
        parquet.read_schema(merged_single_cells).names
    ]
)
# show the modified column count
len(df_for_normalize_and_feature_select_without_outliers.columns)

df_for_normalize_and_feature_select_without_outliers
# -

print("Length of dataset with outliers: ", len(df_for_normalize_and_feature_select))
print(
    "Length of dataset without outliers: ",
    len(df_for_normalize_and_feature_select_without_outliers),
)

# +
parquet_pycytominer_annotated_wo_outliers = (
    f"./{example_plate}_annotated_wo_outliers.parquet"
)

# check if we already have annotated data
if not pathlib.Path(parquet_pycytominer_annotated_wo_outliers).is_file():
    # annotate the data using pycytominer
    pycytominer.annotate(
        profiles=df_for_normalize_and_feature_select_without_outliers,
        # read the platemap directly from AWS S3 related location
        platemap=df_platemap_and_metadata,
        join_on=["Metadata_well_position", "Metadata_Well"],
        output_file=parquet_pycytominer_annotated_wo_outliers,
        output_type="parquet",
    )

# +
parquet_pycytominer_normalized_wo_outliers = (
    f"./{example_plate}_normalized_wo_outliers.parquet"
)

# check if we already have normalized data
if not pathlib.Path(parquet_pycytominer_normalized_wo_outliers).is_file():
    # normalize the data using pcytominer
    df_pycytominer_normalized = pycytominer.normalize(
        profiles=parquet_pycytominer_annotated_wo_outliers,
        features="infer",
        image_features=False,
        meta_features="infer",
        method="standardize",
        samples="Metadata_control_type == 'negcon'",
        output_file=parquet_pycytominer_normalized_wo_outliers,
        output_type="parquet",
    )

# +
parquet_pycytominer_feature_selected_wo_outliers = (
    f"./{example_plate}_feature_select_wo_outliers.parquet"
)

# check if we already have feature selected data
if not pathlib.Path(parquet_pycytominer_feature_selected_wo_outliers).is_file():
    # feature select normalized data using pycytominer
    df_pycytominer_feature_selected = pycytominer.feature_select(
        profiles=parquet_pycytominer_normalized_wo_outliers,
        operation=[
            "variance_threshold",
            "correlation_threshold",
            "blocklist",
            "drop_na_columns",
        ],
        na_cutoff=0,
        output_file=parquet_pycytominer_feature_selected_wo_outliers,
        output_type="parquet",
    )
# -

# calculate UMAP embeddings from data without coSMicQC-detected outliers
embeddings_without_outliers = generate_umap_embeddings(
    df_input=pd.read_parquet(parquet_pycytominer_feature_selected_wo_outliers),
    cols_metadata_to_exclude=all_metadata_cols,
    random_state=0,
)
# show the shape and top values from the embeddings array
print(embeddings_without_outliers.shape)
embeddings_without_outliers[:3]

# plot UMAP for embeddings without outliers
plot_hvplot_scatter(
    embeddings=embeddings_without_outliers,
    title=f"UMAP of JUMP embeddings from {example_plate} (without erroneous outliers)",
    filename=(
        image_without_all_outliers
        := f"./images/umap_without_outliers_{example_plate}.png"
    ),
    bgcolor="white",
    cmap=px.colors.sequential.Greens[4:],
    clabel="density of single cells",
)
# conserve filespace by displaying export instead of dynamic plot
Image(image_without_all_outliers)

# compare the UMAP images with and without outliers side by side
HTML(
    f"""
    <div style="display: flex;">
      <img src="{image_with_all_outliers}" alt="UMAP which includes erroneous outliers" style="width: 50%;"/>
      <img src="{image_without_all_outliers}" alt="UMAP which includes no erroneous outliers" style="width: 50%;"/>
    </div>
    """
)

# +
# concatenate embeddings together
combined_embeddings = np.vstack((embeddings_with_outliers, embeddings_without_outliers))

# Step 2: Create the labels array
combined_labels = np.concatenate(
    [np.zeros(len(embeddings_with_outliers)), np.ones(len(embeddings_without_outliers))]
)

# visualize UMAP embeddings both with and without outliers together for comparison
plot_hvplot_scatter(
    embeddings=combined_embeddings,
    title=f"UMAP comparing JUMP embeddings from {example_plate} with and without erroneous outliers",
    filename=f"./images/umap_comparison_with_and_without_erroneous_outliers_{example_plate}.png",
    color_dataframe=pd.DataFrame(
        combined_labels, columns=["combined_embedding_color_label"]
    ),
    color_column="combined_embedding_color_label",
    bgcolor="white",
    cmap=[
        "#e76f51",  # Darkest Orange
        "#f4a261",  # Darker Orange
        "#ffbb78",  # Light Orange
        "#aec7e8",  # Light Blue
        "#6baed6",  # Darker Blue
        "#1f77b4",  # Darkest Blue
    ],
    clabel="density of single cells with (orange) and without outliers (blue)",
)
