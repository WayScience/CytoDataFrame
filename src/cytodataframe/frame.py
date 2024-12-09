"""
Defines a CytoDataFrame class.
"""

import base64
import pathlib
import re
from collections.abc import Callable
from io import BytesIO, StringIO
from typing import (
    Any,
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
import skimage
import skimage.io
import skimage.measure
from IPython import get_ipython
from pandas._config import (
    get_option,
)
from pandas.io.formats import (
    format as fmt,
)
from skimage.util import img_as_ubyte

from .image import (
    adjust_with_adaptive_histogram_equalization,
    draw_outline_on_image_from_mask,
    draw_outline_on_image_from_outline,
)

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

    def __init__(  # noqa: PLR0913
        self: CytoDataFrame_type,
        data: Union[CytoDataFrame_type, pd.DataFrame, str, pathlib.Path],
        data_context_dir: Optional[str] = None,
        data_bounding_box: Optional[pd.DataFrame] = None,
        data_mask_context_dir: Optional[str] = None,
        data_outline_context_dir: Optional[str] = None,
        segmentation_file_regex: Optional[Dict[str, str]] = None,
        image_adjustment: Optional[Callable] = None,
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
            data_outline_context_dir: Optional[str]:
                Directory context for the outline data for images.
            segmentation_file_regex: Optional[Dict[str, str]]:
                A dictionary which includes regex strings for mapping segmentation
                images (masks or outlines) to unsegmented images.
            image_adjustment: Callable
                A callable function which will be used to make image adjustments
                when they are processed by CytoDataFrame. The function should
                include a single parameter which takes as input a np.ndarray and
                return the same after adjustments. Defaults to None,
                which will incur an adaptive histogram equalization on images.
                Reference histogram equalization for more information:
                https://scikit-image.org/docs/stable/auto_examples/color_exposure/
            **kwargs:
                Additional keyword arguments to pass to the pandas read functions.
        """

        self._custom_attrs = {
            "data_source": None,
            "data_context_dir": (
                data_context_dir if data_context_dir is not None else None
            ),
            "data_bounding_box": None,
            "data_mask_context_dir": (
                data_mask_context_dir if data_mask_context_dir is not None else None
            ),
            "data_outline_context_dir": (
                data_outline_context_dir
                if data_outline_context_dir is not None
                else None
            ),
            "segmentation_file_regex": (
                segmentation_file_regex if segmentation_file_regex is not None else None
            ),
            "image_adjustment": (
                image_adjustment if image_adjustment is not None else None
            ),
        }

        if isinstance(data, CytoDataFrame):
            self._custom_attrs["data_source"] = data._custom_attrs["data_source"]
            self._custom_attrs["data_context_dir"] = data._custom_attrs[
                "data_context_dir"
            ]
            self._custom_attrs["data_mask_context_dir"] = data._custom_attrs[
                "data_mask_context_dir"
            ]
            self._custom_attrs["data_outline_context_dir"] = data._custom_attrs[
                "data_outline_context_dir"
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
                data_outline_context_dir=self._custom_attrs["data_outline_context_dir"],
                segmentation_file_regex=self._custom_attrs["segmentation_file_regex"],
                image_adjustment=self._custom_attrs["image_adjustment"],
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
                data_outline_context_dir=self._custom_attrs["data_outline_context_dir"],
                segmentation_file_regex=self._custom_attrs["segmentation_file_regex"],
                image_adjustment=self._custom_attrs["image_adjustment"],
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

    def search_for_mask_or_outline(  # noqa: PLR0913, PLR0911
        self: CytoDataFrame_type,
        data_value: str,
        pattern_map: dict,
        file_dir: str,
        candidate_path: pathlib.Path,
        orig_image: np.ndarray,
        mask: bool = True,
    ) -> np.ndarray:
        """
        Search for a mask or outline image file based on the
        provided patterns and apply it to the target image.

        Args:
            data_value (str):
                The value used to match patterns for locating
                mask or outline files.
            pattern_map (dict):
                A dictionary of file patterns and their corresponding
                original patterns for matching.
            file_dir (str):
                The directory where image files are stored.
            candidate_path (pathlib.Path):
                The path to the candidate image file to apply
                the mask or outline to.
            orig_image (np.ndarray):
                The image which will have a mask or outline applied.
            mask (bool, optional):
                Whether to search for a mask (True) or an outline (False).
                Default is True.

        Returns:
            np.ndarray:
                The target image with the applied mask or outline,
                or None if no relevant file is found.
        """
        if file_dir is None:
            return None

        if pattern_map is None:
            matching_mask_file = list(
                pathlib.Path(file_dir).rglob(f"{pathlib.Path(candidate_path).stem}*")
            )
            if matching_mask_file:
                if mask:
                    return draw_outline_on_image_from_mask(
                        orig_image=orig_image, mask_image_path=matching_mask_file[0]
                    )
                else:
                    return draw_outline_on_image_from_outline(
                        orig_image=orig_image, outline_image_path=matching_mask_file[0]
                    )
            return None

        for file_pattern, original_pattern in pattern_map.items():
            if re.search(original_pattern, data_value):
                matching_files = [
                    file
                    for file in pathlib.Path(file_dir).rglob("*")
                    if re.search(file_pattern, file.name)
                ]
                if matching_files:
                    if mask:
                        return draw_outline_on_image_from_mask(
                            orig_image=orig_image, mask_image_path=matching_files[0]
                        )
                    else:
                        return draw_outline_on_image_from_outline(
                            orig_image=orig_image, outline_image_path=matching_files[0]
                        )

        return None

    def process_image_data_as_html_display(
        self: CytoDataFrame_type,
        data_value: Any,  # noqa: ANN401
        bounding_box: Tuple[int, int, int, int],
    ) -> str:
        """
        Process the image data based on the provided data value
        and bounding box, applying masks or outlines where
        applicable, and return an HTML representation of the
        cropped image for display.

        Args:
            data_value (Any):
                The value to search for in the file system or as the image data.
            bounding_box (Tuple[int, int, int, int]):
                The bounding box to crop the image.

        Returns:
            str:
                The HTML image display string, or the unmodified data
                value if the image cannot be processed.
        """
        candidate_path = None
        # Get the pattern map for segmentation file regex
        pattern_map = self._custom_attrs.get("segmentation_file_regex")

        # Step 1: Find the candidate file if the data value is not already a file
        if not pathlib.Path(data_value).is_file():
            # Search for the data value in the data context directory
            if candidate_paths := list(
                pathlib.Path(self._custom_attrs["data_context_dir"]).rglob(data_value)
            ):
                # If a candidate file is found, use the first one
                candidate_path = candidate_paths[0]
                orig_image_array = skimage.io.imread(candidate_path)

                # Adjust the image with image adjustment callable
                # or adaptive histogram equalization
                if self._custom_attrs["image_adjustment"] is not None:
                    orig_image_array = self._custom_attrs["image_adjustment"](
                        orig_image_array
                    )
                else:
                    orig_image_array = adjust_with_adaptive_histogram_equalization(
                        orig_image_array
                    )

                # Normalize to 0-255 for image saving
                orig_image_array = img_as_ubyte(orig_image_array)

            else:
                # If no candidate file is found, return the original data value
                return data_value

        prepared_image = None
        # Step 2: Search for a mask
        prepared_image = self.search_for_mask_or_outline(
            data_value=data_value,
            pattern_map=pattern_map,
            file_dir=self._custom_attrs["data_mask_context_dir"],
            candidate_path=candidate_path,
            orig_image=orig_image_array,
            mask=True,
        )

        # If no mask is found, proceed to search for an outline
        if prepared_image is None:
            # Step 3: Search for an outline if no mask was found
            prepared_image = self.search_for_mask_or_outline(
                data_value=data_value,
                pattern_map=pattern_map,
                file_dir=self._custom_attrs["data_outline_context_dir"],
                candidate_path=candidate_path,
                orig_image=orig_image_array,
                mask=False,
            )

        # Step 4: If neither mask nor outline is found, use the original image array
        if prepared_image is None:
            prepared_image = orig_image_array

        # Step 5: Crop the image based on the bounding box and encode it to PNG format
        try:
            x_min, y_min, x_max, y_max = map(int, bounding_box)  # Ensure integers
            cropped_img_array = prepared_image[
                y_min:y_max, x_min:x_max
            ]  # Perform slicing
        except ValueError as e:
            raise ValueError(
                f"Bounding box contains invalid values: {bounding_box}"
            ) from e
        except IndexError as e:
            raise IndexError(
                f"Bounding box {bounding_box} is out of bounds for image dimensions "
                f"{prepared_image.shape}"
            ) from e

        # Step 6:
        try:
            # Save cropped image to buffer
            png_bytes_io = BytesIO()
            skimage.io.imsave(
                png_bytes_io, cropped_img_array, plugin="imageio", extension=".png"
            )
            png_bytes = png_bytes_io.getvalue()

        except (FileNotFoundError, ValueError) as exc:
            # Handle errors if image processing fails
            print(exc)
            return data_value

        # Return HTML image display as a base64-encoded PNG
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
