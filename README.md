
# Survey Subsampling

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![stability-wip](https://img.shields.io/badge/stability-work_in_progress-lightgrey.svg)
[![L-GPL License](https://img.shields.io/badge/license-L--GPL-blue.svg)](https://github.com/childmindresearch/survey_subsampling/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/survey_subsampling)

This is the beginnings of a package which takes a data-driven approach to reducing the size of complex survey instruments based on their ability to predict independently collected assessment data. In simpler words, reduce tools like the CBCL to fewer items, based on their ability to predict diagnostic status.

## Approach

The approach is quite simple. We:

1. Prune the dataset to include only high-prevalance categories
1. Learn the relationships between the complete assessment battery and the target outcomes
1. Further prune the dataset to only continue exploration on models with reasonable best-case models
1. Sort assessment items based on their relative importance in predicting the outcome
1. Re-learn relationships while incrementally removing the least important assessment items

## Installation

Install this package via :

```bash
pip install git+https://github.com/cmi-dair/survey_subsampling
```

## Quick start

There are a few scripts in this library so far, and there are specific usecases for each of them.

### Have `.Rdata` that you'd like to work with?

```bash
subsample_convert <your_Rdata_file> <your_converted_parquet_file>
```

### Want to subsample your assessment instrument?

```bash
subsample <your_parquet_file> <your_output_directory>  # And optionally, a bunch of other arguments
```


## Notes

Currently, there are many limitations in this module. In particular:
- The CBCL, ABCL, and their harmonized items are hard-coded
- The diagnostic labels are hard-coded
- This is has solely been tested on the Brazillian High Risk Cohort dataset, and its quirks of organization
- Plotting functionality is currently coupled with the evaluation
- Plotting functionality for the degraded instrument evaluation has not been ported from the notebook
- Subject-specific effects have not been modelled or accounted for
- Many more, I'm sure