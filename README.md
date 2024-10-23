# CytoDataFrame

[![PyPI - Version](https://img.shields.io/pypi/v/cytodataframe)](https://pypi.org/project/CytoDataFrame/)
[![Build Status](https://github.com/WayScience/CytoDataFrame/actions/workflows/run-tests.yml/badge.svg?branch=main)](https://github.com/WayScience/CytoDataFrame/actions/workflows/run-tests.yml?query=branch%3Amain)
![Coverage Status](https://raw.githubusercontent.com/WayScience/CytoDataFrame/main/media/coverage-badge.svg)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)

![](https://raw.githubusercontent.com/WayScience/coSMicQC/refs/heads/main/docs/presentations/2024-09-18-SBI2-Conference/images/cosmicqc-example-cytodataframe.png)
_CytoDataFrame extends Pandas functionality to help display single-cell profile data alongside related images._

CytoDataFrame is an advanced in-memory data analysis format designed for single-cell profiling, integrating not only the data profiles but also their corresponding microscopy images and segmentation masks.
Traditional single-cell profiling often excludes the associated images from analysis, limiting the scope of research.
CytoDataFrame bridges this gap, offering a purpose-built solution for comprehensive analysis that incorporates both the data and images, empowering more detailed and visual insights in single-cell research.

CytoDataFrame development began within [coSMicQC](https://github.com/WayScience/coSMicQC) - a single-cell profile quality control package.

## Installation

Install CytoDataFrame from source using the following:

```shell
# install from pypi
pip install cytodataframe

# or install directly from source
pip install git+https://github.com/WayScience/CytoDataFrame.git
```

## Contributing, Development, and Testing

Please see our [contributing](https://WayScience.github.io/CytoDataFrame/main/contributing) documentation for more details on contributions, development, and testing.

## References

- [coSMicQC](https://github.com/WayScience/coSMicQC)
- [pycytominer](https://github.com/cytomining/pycytominer)
- [CellProfiler](https://github.com/CellProfiler/CellProfiler)
- [CytoTable](https://github.com/cytomining/CytoTable)
