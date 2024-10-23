<img height="200" src="https://raw.githubusercontent.com/WayScience/CytoDataFrame/main/media/logo/with-text-for-light-bg.png?raw=true">

# Single cell Morphology Quality Control

![PyPI - Version](https://img.shields.io/pypi/v/CytoDataFrame)
[![Build Status](https://github.com/WayScience/CytoDataFrame/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/WayScience/CytoDataFrame/actions/workflows/run-tests.yml?query=branch%3Amain)
![Coverage Status](https://raw.githubusercontent.com/WayScience/CytoDataFrame/main/media/coverage-badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

> 🌠 Navigate the cosmos of single-cell morphology with confidence — CytoDataFrame keeps your data on course!

CytoDataFrame is a Python package to evaluate converted single-cell morphology outputs from CytoTable.

It can be challenging to get "perfect" single-cell compartment segmentation across large high-throughput screens when performing object detection in CellProfiler (or similar software).
Technical artifacts can arise during segmentation, leading to issues such as under-segmentation, over-segmentation, or the erroneous segmentation of background noise, smudges, or bright artifacts.

In single-cell analysis, intriguing phenotypes often emerge by examining morphological differences.
However, technical outliers in the data can obscure these biological insights, compromising the validity of the findings.

By utilizing specific morphological features extracted with CellProfiler, particularly AreaShape features, you can identify technically incorrect segmentations.
These can then be labeled or removed before further preprocessing steps, such as those performed with pycytominer.

## Installation

Install CytoDataFrame from [PyPI](https://pypi.org/project/CytoDataFrame/) or from source:

```shell
# install from pypi
pip install CytoDataFrame

# install directly from source
pip install git+https://github.com/WayScience/CytoDataFrame.git
```

## Contributing, Development, and Testing

Please see our [contributing](https://WayScience.github.io/CytoDataFrame/main/contributing) documentation for more details on contributions, development, and testing.

## References

- [pycytominer](https://github.com/cytomining/pycytominer)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [CytoTable](https://github.com/cytomining/CytoTable)
