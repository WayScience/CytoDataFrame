"""
Defines a CytoDataFrame class.
"""

import base64
import pathlib
import random
import re
import webbrowser
from io import BytesIO, StringIO
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
import plotly
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import skimage
import skimage.io
import skimage.measure
from IPython import get_ipython
from jinja2 import Environment, FileSystemLoader
from pandas._config import (
    get_option,
)
from pandas.io.formats import (
    format as fmt,
)
from PIL import Image, ImageDraw

from .image import adjust_image_brightness, is_image_too_dark

# provide backwards compatibility for Self type in earlier Python versions.
# see: https://peps.python.org/pep-0484/#annotating-instance-and-class-methods
CytoDataFrame_type = TypeVar("CytoDataFrame_type", bound="CytoDataFrame")


class CytoDataFrame(pd.DataFrame):
    """
    A class designed to enhance single-cell data handling by wrapping
    pandas DataFrame capabilities, providing advanced methods for quality control,
    comprehensive analysis, and image-based data processing.

    This class can initialize with either a pandas DataFrame or a file path (CSV, TSV,
    TXT, or Parquet). When initialized with a file path, it reads the data into a
    pandas DataFrame. It also includes capabilities to export data.

    Attributes:
        _metadata (ClassVar[list[str]]):
            A class-level attribute that includes custom attributes.
        _custom_attrs (dict):
            A dictionary to store custom attributes, such as data source,
            context directory, and bounding box information.
    """

    _metadata: ClassVar = ["_custom_attrs"]

    def __init__(  # noqa: PLR0912
        self: CytoDataFrame_type,
        data: Union[CytoDataFrame_type, pd.DataFrame, str, pathlib.Path],
        data_context_dir: Optional[str] = None,
        data_bounding_box: Optional[pd.DataFrame] = None,
        data_mask_context_dir: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Initializes the CytoDataFrame with either a DataFrame or a file path.

        Args:
            data (Union[CytoDataFrame_type, pd.DataFrame, str, pathlib.Path]):
                The data source, either a pandas DataFrame or a file path.
            data_context_dir (Optional[str]):
                Directory context for the image data within the DataFrame.
            data_bounding_box (Optional[pd.DataFrame]):
                Bounding box data for the DataFrame images.
            data_mask_context_dir: Optional[str]:
                Directory context for the mask data for images.
            **kwargs:
                Additional keyword arguments to pass to the pandas read functions.
        """

        self._custom_attrs = {
            "data_source": None,
            "data_context_dir": None,
            "data_bounding_box": None,
            "data_mask_context_dir": None,
        }

        if data_context_dir is not None:
            self._custom_attrs["data_context_dir"] = data_context_dir

        if data_mask_context_dir is not None:
            self._custom_attrs["data_mask_context_dir"] = data_mask_context_dir

        if isinstance(data, CytoDataFrame):
            self._custom_attrs["data_source"] = data._custom_attrs["data_source"]
            self._custom_attrs["data_context_dir"] = data._custom_attrs[
                "data_context_dir"
            ]
            self._custom_attrs["data_mask_context_dir"] = data._custom_attrs[
                "data_mask_context_dir"
            ]
            super().__init__(data)
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            self._custom_attrs["data_source"] = (
                "pandas.DataFrame"
                if isinstance(data, pd.DataFrame)
                else "pandas.Series"
            )
            super().__init__(data)
        elif isinstance(data, (str, pathlib.Path)):
            data_path = pathlib.Path(data)
            self._custom_attrs["data_source"] = str(data_path)

            if data_context_dir is None:
                self._custom_attrs["data_context_dir"] = str(data_path.parent)
            else:
                self._custom_attrs["data_context_dir"] = data_context_dir

            if data_path.suffix in {".csv", ".tsv", ".txt"} or data_path.suffixes == [
                ".csv",
                ".gz",
            ]:
                data = pd.read_csv(data_path, **kwargs)
            elif data_path.suffix == ".parquet":
                data = pd.read_parquet(data_path, **kwargs)
            else:
                raise ValueError("Unsupported file format for CytoDataFrame.")

            super().__init__(data)

        else:
            super().__init__(data)

        if data_bounding_box is None:
            self._custom_attrs["data_bounding_box"] = self.get_bounding_box_from_data()

        else:
            self._custom_attrs["data_bounding_box"] = data_bounding_box

    def __getitem__(self: CytoDataFrame_type, key: Union[int, str]) -> Any:  # noqa: ANN401
        """
        Returns an element or a slice of the underlying pandas DataFrame.

        Args:
            key:
                The key or slice to access the data.

        Returns:
            pd.DataFrame or any:
                The selected element or slice of data.
        """

        result = super().__getitem__(key)

        if isinstance(result, pd.Series):
            return result

        elif isinstance(result, pd.DataFrame):
            return CytoDataFrame(
                super().__getitem__(key),
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
                data_mask_context_dir=self._custom_attrs["data_mask_context_dir"],
            )

    def _wrap_method(
        self: CytoDataFrame_type,
        method: Callable,
        *args: List[Any],
        **kwargs: Dict[str, Any],
    ) -> Any:  # noqa: ANN401
        """
        Wraps a given method to ensure that the returned result
        is an CytoDataFrame if applicable.

        Args:
            method (Callable):
                The method to be called and wrapped.
            *args (List[Any]):
                Positional arguments to be passed to the method.
            **kwargs (Dict[str, Any]):
                Keyword arguments to be passed to the method.

        Returns:
            Any:
                The result of the method call. If the result is a pandas DataFrame,
                it is wrapped in an CytoDataFrame instance with additional context
                information (data context directory and data bounding box).

        """
        result = method(*args, **kwargs)
        if isinstance(result, pd.DataFrame):
            result = CytoDataFrame(
                result,
                data_context_dir=self._custom_attrs["data_context_dir"],
                data_bounding_box=self._custom_attrs["data_bounding_box"],
                data_mask_context_dir=self._custom_attrs["data_mask_context_dir"],
            )
        return result

    def sort_values(
        self: CytoDataFrame_type, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> CytoDataFrame_type:
        """
        Sorts the DataFrame by the specified column(s) and returns a
        new CytoDataFrame instance.

        Note: we wrap this method within CytoDataFrame to help ensure the consistent
        return of CytoDataFrames in the context of pd.Series (which are
        treated separately but have specialized processing within the
        context of sort_values).

        Args:
            *args (List[Any]):
                Positional arguments to be passed to the pandas
                DataFrame's `sort_values` method.
            **kwargs (Dict[str, Any]):
                Keyword arguments to be passed to the pandas
                DataFrame's `sort_values` method.

        Returns:
            CytoDataFrame_type:
                A new instance of CytoDataFrame sorted by the specified column(s).

        """

        return self._wrap_method(super().sort_values, *args, **kwargs)

    def get_bounding_box_from_data(
        self: CytoDataFrame_type,
    ) -> Optional[CytoDataFrame_type]:
        """
        Retrieves bounding box data from the DataFrame based
        on predefined column groups.

        This method identifies specific groups of columns representing bounding box
        coordinates for different cellular components (cytoplasm, nuclei, cells) and
        checks for their presence in the DataFrame. If all required columns are present,
        it filters and returns a new CytoDataFrame instance containing only these
        columns.

        Returns:
            Optional[CytoDataFrame_type]:
                A new instance of CytoDataFrame containing the bounding box columns if
                they exist in the DataFrame. Returns None if the required columns
                are not found.

        """
        # Define column groups and their corresponding conditions
        column_groups = {
            "cyto": [
                "Cytoplasm_AreaShape_BoundingBoxMaximum_X",
                "Cytoplasm_AreaShape_BoundingBoxMaximum_Y",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_X",
                "Cytoplasm_AreaShape_BoundingBoxMinimum_Y",
            ],
            "nuclei": [
                "Nuclei_AreaShape_BoundingBoxMaximum_X",
                "Nuclei_AreaShape_BoundingBoxMaximum_Y",
                "Nuclei_AreaShape_BoundingBoxMinimum_X",
                "Nuclei_AreaShape_BoundingBoxMinimum_Y",
            ],
            "cells": [
                "Cells_AreaShape_BoundingBoxMaximum_X",
                "Cells_AreaShape_BoundingBoxMaximum_Y",
                "Cells_AreaShape_BoundingBoxMinimum_X",
                "Cells_AreaShape_BoundingBoxMinimum_Y",
            ],
        }

        # Determine which group of columns to select based on availability in self.data
        selected_group = None
        for group, cols in column_groups.items():
            if all(col in self.columns.tolist() for col in cols):
                selected_group = group
                break

        # Assign the selected columns to self.bounding_box_df
        if selected_group:
            return self.filter(items=column_groups[selected_group])

        return None

    def export(
        self: CytoDataFrame_type, file_path: str, **kwargs: Dict[str, Any]
    ) -> None:
        """
        Exports the underlying pandas DataFrame to a file.

        Args:
            file_path (str):
                The path where the DataFrame should be saved.
            **kwargs:
                Additional keyword arguments to pass to the pandas to_* methods.
        """

        data_path = pathlib.Path(file_path)

        # export to csv
        if ".csv" in data_path.suffixes:
            self.to_csv(file_path, **kwargs)
        # export to tsv
        elif any(elem in data_path.suffixes for elem in (".tsv", ".txt")):
            self.to_csv(file_path, sep="\t", **kwargs)
        # export to parquet
        elif data_path.suffix == ".parquet":
            self.to_parquet(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format for export.")

    @staticmethod
    def is_notebook_or_lab() -> bool:
        """
        Determines if the code is being executed in a Jupyter notebook (.ipynb)
        returning false if it is not.

        This method attempts to detect the interactive shell environment
        using IPython's `get_ipython` function. It checks the class name of the current
        IPython shell to distinguish between different execution environments.

        Returns:
            bool:
                - `True`
                    if the code is being executed in a Jupyter notebook (.ipynb).
                - `False`
                    otherwise (e.g., standard Python shell, terminal IPython shell,
                    or scripts).
        """
        try:
            # check for type of session via ipython
            shell = get_ipython().__class__.__name__
            if "ZMQInteractiveShell" in shell:
                return True
            elif "TerminalInteractiveShell" in shell:
                return False
            else:
                return False
        except NameError:
            return False

    def show_report(
        self: CytoDataFrame_type,
        report_path: Optional[str] = None,
        auto_open: bool = True,
        color_palette: Optional[List[str]] = None,
    ) -> None:
        """
        Generates and displays a report based on the current DataFrame's data
        quality control (DQC) columns.

        This method organizes the DQC columns from the DataFrame, creates
        visualizations for each threshold set, and then either displays the
        visualizations inline (if running in a Jupyter notebook or lab) or
        opens an HTML report in the default web browser.

        Args:
            report_path (Optional[str]):
                The file path where the HTML report should be saved and displayed.
                If `None`, the report will be displayed inline if in a notebook
                or lab environment.
            auto_open: bool:
                Whether to automatically open the report.
            color_palette Optional(List[str]):
                Optional list for color palette to use.

        Raises:
            ValueError: If the DataFrame does not contain any DQC columns.
        """

        # find all cosmicqc columns in the data using the prefix `cqc.`
        cqc_cols = [col for col in self.columns.tolist() if "cqc." in col]
        # organize column data into the threshold set name, threshold is_outlier col,
        # and the threshold score columns as list
        organized_columns = [
            [
                # name of the threshold set
                threshold_set,
                # column which includes boolean is_outlier data for threshold set
                next(
                    (
                        col
                        for col in cqc_cols
                        if f"cqc.{threshold_set}.is_outlier" in col
                    ),
                    None,
                ),
                # columns which show the data associated with thresholds
                [col for col in cqc_cols if f"cqc.{threshold_set}.Z_Score." in col],
            ]
            for threshold_set in sorted({col.split(".")[1] for col in cqc_cols})
        ]

        # create figures for visualization based on the name, outlier status,
        # and threshold scores
        figures = [
            self.create_threshold_set_outlier_visualization(
                df=self,
                threshold_set_name=set_name,
                col_outlier=col_outlier,
                cols_threshold_scores=cols_threshold_scores,
                color_palette=color_palette,
            )
            for set_name, col_outlier, cols_threshold_scores in organized_columns
        ]

        # if we're running in a notebook or jupyter lab, show the figures as-is
        if self.is_notebook_or_lab() or report_path is None:
            # if we should automatically open, show the figures
            if auto_open:
                for figure in figures:
                    figure.show()

            return figures

        # otherwise, create an html file with figures and open it with default browser
        else:
            html_path = self.create_figure_group_html(
                figures=figures, report_path=report_path
            )

            # if we should auto open, show the html file in default web browser
            if auto_open:
                webbrowser.open(f"file://{pathlib.Path(html_path).resolve()}")

            print(f"Opened default web browser for report {html_path}")

            return html_path

    @staticmethod
    def create_figure_group_html(
        figures: List[plotly.graph_objs._figure.Figure],
        report_path: Optional[str] = None,
    ) -> str:
        """
        Generates an HTML file containing multiple Plotly figures.

        This method takes a list of Plotly figure objects, converts them to HTML,
        and embeds them into a template HTML file. The resulting HTML file is then
        saved to the specified path.

        Args:
            figures (List[plotly.graph_objs._figure.Figure]):
                A list of Plotly figure objects to be included in the HTML report.
            report_path (str):
                The file path where the HTML report will be saved.
                Defaults to "cosmicqc_outlier_report.html" when None.

        Returns:
            str: The path to the saved HTML report.
        """

        # if we have none for the report path, use a default name.
        if report_path is None:
            report_path = "cosmicqc_outlier_report.html"

        # create wrapped html for figures
        figure_html = "".join(
            [
                f"<div class='fig_wrapper'>{fig.to_html(full_html=False)}</div>"
                for fig in figures
            ]
        )

        # configure jinja environment
        env = Environment(
            loader=FileSystemLoader(f"{pathlib.Path(__file__).parent!s}/data")
        )
        # load a jinja template
        template = env.get_template("report_template.html")

        # Render the template with Plotly figure HTML
        rendered_html = template.render(figure_html=figure_html)

        # write the html to file
        with open(report_path, "w") as f:
            f.write(rendered_html)

        # return the path of the file
        return report_path

    def create_threshold_set_outlier_visualization(
        self: CytoDataFrame_type,
        df: pd.DataFrame,
        threshold_set_name: str,
        col_outlier: str,
        cols_threshold_scores: List[str],
        color_palette: Optional[List[str]] = None,
    ) -> plotly.graph_objs._figure.Figure:
        """
        Creates a Plotly figure visualizing the Z-score distributions and outliers
        for a given threshold set.

        This method generates histograms for each Z-score column in the given DataFrame,
        colors them based on outlier status, and overlays them into a single figure.

        Args:
            df (pd.DataFrame):
                The DataFrame containing the data to be visualized.
            threshold_set_name (str):
                The name of the threshold set being visualized.
            col_outlier (str):
                The column name indicating outlier status.
            cols_threshold_scores (List[str]):
                A list of column names representing the Z-scores to be visualized.
            color_palette Optional(List[str]):
                Optional list for color palette to use.
                Defaults to use Dark24 color palette from Plotly.

        Returns:
            plotly.graph_objs._figure.Figure:
                A Plotly figure object containing the visualization.
        """

        # Create histograms using plotly.express with pattern_shape and random color
        figures = [
            px.histogram(
                df,
                x=col,
                color=col_outlier,
                nbins=50,
                pattern_shape=col_outlier,
                opacity=0.7,
            )
            for col in cols_threshold_scores
        ]

        # Create a combined figure
        fig = go.Figure()

        # check that we have enough colors for figures if provided
        if color_palette is not None and len(color_palette) < len(figures):
            raise ReferenceError(
                f"Color palette length must match figure length of {len(figures)}."
            )

        # Add traces from each histogram and modify colors, names, and pattern shapes
        for idx, fig_hist in enumerate(figures):
            if color_palette is None:
                # Create a default list of colors from a Plotly color palette
                fig_color = random.choice(pc.qualitative.Dark24)
            else:
                # otherwise, use static color palette based on the number of figures
                fig_color = color_palette[idx]

            for trace in fig_hist.data:
                trace.marker.color = fig_color
                trace.marker.pattern.shape = (
                    "x" if trace.name == "True" else ""
                )  # Use pattern shapes
                renamed_col = cols_threshold_scores[idx].replace(
                    f"cqc.{threshold_set_name}.Z_Score.", ""
                )
                trace.name = (
                    f"{renamed_col} ({'outlier' if trace.name == 'True' else 'inlier'})"
                )
                # Update hovertemplate to match the name in the key
                trace.hovertemplate = (
                    f"<b>{renamed_col}</b><br>"
                    + "Z-Score: %{x}<br>"
                    + "Single-cell Count (log): %{y}<br>"
                    + "<extra></extra>"
                )
                fig.add_trace(trace)

        # Update layout
        fig.update_layout(
            title=f"{threshold_set_name.replace('_', ' ').title()} Z-Score Outliers",
            xaxis_title="Z-Score",
            yaxis_title="Single-cell Count (log)",
            yaxis_type="log",
            # ensures that histograms are overlapping
            barmode="overlay",
            legend_title_text="Measurement Type and QC Status",
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 0.95,
                "xanchor": "left",
                "x": 1.02,
            },
        )

        return fig

    def find_image_columns(self: CytoDataFrame_type) -> bool:
        pattern = r".*\.(tif|tiff)$"
        return [
            column
            for column in self.columns
            if self[column]
            .apply(
                lambda value: isinstance(value, str)
                and re.match(pattern, value, flags=re.IGNORECASE)
            )
            .any()
        ]

    @staticmethod
    def draw_outline_on_image(actual_image_path: str, mask_image_path: str) -> Image:
        """
        Draws outlines on a TIFF image based on a mask image and returns
        the combined result.

        This method takes the path to a TIFF image and a mask image, creates
        an outline from the mask, and overlays it on the TIFF image. The resulting
        image, which combines the TIFF image with the mask outline, is returned.

        Args:
            actual_image_path (str): Path to the TIFF image file.
            mask_image_path (str): Path to the mask image file.

        Returns:
            PIL.Image.Image: A PIL Image object that is the result of
            combining the TIFF image with the mask outline.

        Raises:
            FileNotFoundError: If the specified image or mask file does not exist.
            ValueError: If the images are not in compatible formats or sizes.
        """
        # Load the TIFF image
        tiff_image_array = skimage.io.imread(actual_image_path)

        # Check if the image is 16-bit and grayscale
        if tiff_image_array.dtype == np.uint16:
            # Normalize the image to 8-bit for display purposes
            tiff_image_array = (tiff_image_array / 256).astype(np.uint8)

        # Convert to PIL Image and then to 'RGBA'
        tiff_image = Image.fromarray(tiff_image_array).convert("RGBA")

        # Check if the image is too dark and adjust brightness if needed
        if is_image_too_dark(tiff_image):
            tiff_image = adjust_image_brightness(tiff_image)

        # Load the mask image and convert it to grayscale
        mask_image = Image.open(mask_image_path).convert("L")
        mask_array = np.array(mask_image)
        mask_array[mask_array > 0] = 255  # Ensure non-zero values are 255 (white)

        # Find contours of the mask
        contours = skimage.measure.find_contours(mask_array, level=128)

        # Create an outline image with transparent background
        outline_image = Image.new("RGBA", tiff_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(outline_image)

        for contour in contours:
            # Swap x and y to match image coordinates
            draw.line(
                [(x, y) for y, x in np.round(contour).astype(int)],
                fill=(0, 255, 0, 200),
                width=2,
            )

        # Combine the TIFF image with the outline image
        return Image.alpha_composite(tiff_image, outline_image)

    def process_image_data_as_html_display(
        self: CytoDataFrame_type,
        data_value: Any,  # noqa: ANN401
        bounding_box: Tuple[int, int, int, int],
    ) -> str:
        if not pathlib.Path(data_value).is_file():
            # Use rglob to recursively search for a matching file
            if candidate_paths := list(
                pathlib.Path(self._custom_attrs["data_context_dir"]).rglob(data_value)
            ):
                # if we find a candidate, return the first one
                candidate_path = candidate_paths[0]
            else:
                # we don't have any candidate paths so return the unmodified value
                return data_value

        try:
            if self._custom_attrs["data_mask_context_dir"] is not None and (
                matching_mask_file := list(
                    pathlib.Path(self._custom_attrs["data_mask_context_dir"]).rglob(
                        f"{pathlib.Path(candidate_path).stem}*"
                    )
                )
            ):
                pil_image = self.draw_outline_on_image(
                    actual_image_path=candidate_path,
                    mask_image_path=matching_mask_file[0],
                )

            else:
                # Read the TIFF image from the byte array
                tiff_image = skimage.io.imread(candidate_path)

                # Convert the image array to a PIL Image
                pil_image = Image.fromarray(tiff_image)

            cropped_img = pil_image.crop(bounding_box)

            # Save the PIL Image as PNG to a BytesIO object
            png_bytes_io = BytesIO()
            cropped_img.save(png_bytes_io, format="PNG")

            # Get the PNG byte data
            png_bytes = png_bytes_io.getvalue()

        except (FileNotFoundError, ValueError):
            # return the raw data value if we run into an exception of some kind
            print("Unable to process image from {candidate_path}")
            return data_value

        return (
            '<img src="data:image/png;base64,'
            f'{base64.b64encode(png_bytes).decode("utf-8")}" style="width:300px;"/>'
        )

    def get_displayed_rows(self: CytoDataFrame_type) -> List[int]:
        # Get the current display settings
        max_rows = pd.get_option("display.max_rows")
        min_rows = pd.get_option("display.min_rows")

        if len(self) <= max_rows:
            # If the DataFrame has fewer rows than max_rows, all rows will be displayed
            return self.index.tolist()
        else:
            # Calculate how many rows will be displayed at the beginning and end
            half_min_rows = min_rows // 2
            start_display = self.index[:half_min_rows].tolist()
            end_display = self.index[-half_min_rows:].tolist()
            return start_display + end_display

    def _repr_html_(
        self: CytoDataFrame_type, key: Optional[Union[int, str]] = None
    ) -> str:
        """
        Returns HTML representation of the underlying pandas DataFrame
        for use within Juypyter notebook environments and similar.

        Referenced with modifications from:
        https://github.com/pandas-dev/pandas/blob/v2.2.2/pandas/core/frame.py#L1216

        Modifications added to help achieve image-based output for single-cell data
        within the context of CytoDataFrame and coSMicQC.

        Mainly for Jupyter notebooks.

        Returns:
            str: The data in a pandas DataFrame.
        """

        if self._info_repr():
            buf = StringIO()
            self.info(buf=buf)
            # need to escape the <class>, should be the first line.
            val = buf.getvalue().replace("<", r"&lt;", 1)
            val = val.replace(">", r"&gt;", 1)
            return f"<pre>{val}</pre>"

        if get_option("display.notebook_repr_html"):
            max_rows = get_option("display.max_rows")
            min_rows = get_option("display.min_rows")
            max_cols = get_option("display.max_columns")
            show_dimensions = get_option("display.show_dimensions")

            # determine if we have image_cols to display
        if image_cols := self.find_image_columns():
            # re-add bounding box cols if they are no longer available as in cases
            # of masking or accessing various pandas attr's
            bounding_box_externally_joined = False

            if self._custom_attrs["data_bounding_box"] is not None and not all(
                col in self.columns.tolist()
                for col in self._custom_attrs["data_bounding_box"].columns.tolist()
            ):
                data = self.join(other=self._custom_attrs["data_bounding_box"])
                bounding_box_externally_joined = True
            else:
                data = self.copy()

            # gather indices which will be displayed based on pandas configuration
            display_indices = self.get_displayed_rows()

            # gather bounding box columns for use below
            bounding_box_cols = self._custom_attrs["data_bounding_box"].columns.tolist()

            for image_col in image_cols:
                data.loc[display_indices, image_col] = data.loc[display_indices].apply(
                    lambda row: self.process_image_data_as_html_display(
                        data_value=row[image_col],
                        bounding_box=(
                            # rows below are specified using the column name to
                            # determine which part of the bounding box the columns
                            # relate to (the list of column names could be in
                            # various order).
                            row[
                                next(
                                    col
                                    for col in bounding_box_cols
                                    if "Minimum_X" in col
                                )
                            ],
                            row[
                                next(
                                    col
                                    for col in bounding_box_cols
                                    if "Minimum_Y" in col
                                )
                            ],
                            row[
                                next(
                                    col
                                    for col in bounding_box_cols
                                    if "Maximum_X" in col
                                )
                            ],
                            row[
                                next(
                                    col
                                    for col in bounding_box_cols
                                    if "Maximum_Y" in col
                                )
                            ],
                        ),
                    ),
                    axis=1,
                )

            if bounding_box_externally_joined:
                data = data.drop(
                    self._custom_attrs["data_bounding_box"].columns.tolist(), axis=1
                )

            formatter = fmt.DataFrameFormatter(
                data,
                columns=None,
                col_space=None,
                na_rep="NaN",
                formatters=None,
                float_format=None,
                sparsify=None,
                justify=None,
                index_names=True,
                header=True,
                index=True,
                bold_rows=True,
                # note: we avoid escapes to allow HTML rendering for images
                escape=False,
                max_rows=max_rows,
                min_rows=min_rows,
                max_cols=max_cols,
                show_dimensions=show_dimensions,
                decimal=".",
            )

            return fmt.DataFrameRenderer(formatter).to_html()

        else:
            return None
