# Copyright Search Scripts

This directory contains scripts for searching copyright-related information in image metadata across sharded dataframes.

## Overview

The scripts perform copyright detection through two main approaches:
1. EXIF metadata search (`exif_search.py`)
2. Regex pattern matching on captions and OCR text (`regex_search.py`)

The search is performed on sharded dataframes for memory efficiency, with results later combined using `combine_exif_search.py` and `combine_regex_search.py`.

## Script Details

### exif_search.py
- Searches for copyright information in image EXIF metadata
- Looks for copyright-related keys and extracts associated text
- Adds columns:
  - `exif_copyright_match`: Boolean indicating if copyright info found
  - `exif_copyright_text`: Extracted copyright text
  - `exif_copyright_key`: EXIF key containing copyright info

### regex_search.py  
- Performs regex pattern matching on caption and OCR text fields
- Searches for various copyright patterns (©, (c), "all rights reserved", CC licenses, etc.)
- Adds columns for each pattern type:
  - `{pattern}_caption_match`: Match in caption
  - `{pattern}_ocr_text_match`: Match in OCR text
  - `copyright_match`: Overall match indicator

### combine_exif_search.py
- Combines results from sharded EXIF searches
- Creates consolidated `exif_match_subset.parquet` containing only matching records

### combine_regex_search.py
- Combines results from sharded regex pattern searches
- Creates consolidated `copyright_match_subset.parquet` containing only matching records

## Workflow

1. Input data is sharded across multiple directories
2. Search scripts run on individual shards, producing metadata files with match indicators
3. Combine scripts merge results from all shards into single output files
4. Final output includes original metadata plus new copyright-related columns

The sharded approach allows processing of large datasets that wouldn't fit in memory while maintaining efficient search capabilities.

## Example Usage
* `scripts/regex_search.sh` is an example script for running the regex search on Slurm cluster with an array of jobs.
