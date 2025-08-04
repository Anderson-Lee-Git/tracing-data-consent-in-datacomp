# Overview
This repository contains the code for ``How do data owners say no? A case study of data consent mechanisms in
web-scraped vision-language AI training datasets'' (Preprint Upcoming)

## Environment
1. `environment.yml` outlines the dependencies for the main project
2. `watermark_detection` has a separate `environment.yml` in the subdirectory

## Usage
### Copyright Search
`copyright_search` contains the code for 
* Searching copyright information on Caption and OCR texts
* Searching EXIF metadata's copyright fields

The search is done in a sharded fashion, where the data is prepared into shard subdirectories the same way as the default `datacomp` provided. Each search generates a result for each shard. After all shards are prepared, the aggregation script gather the results and save the selected samples into a combined `parquet` file.

### Domain Information
We include the code for the following
1. Extracting and counting the `base_domain` and `full_domain`
    * This is done in sharded fashion as well.
2. Extracting and parsing the `robots.txt` file

We also include the extracted `robots.txt` files under `domain_information/robots_txt` directory, and the annotation for the top 50 *base domains* from `small-en` scale under `domain_information/annotation` directory.

### Watermark Detection
See `tracing-data-consent-in-datacomp/watermark_detection/README.md` for more details.

### Environment Variable
We use `python-dotenv` to manage the environment variables. Therefore, you need to create a `.env` file in the root directory as follows:

```
# Watermark detection
WATERMARK_DETECTION_YOLO_RESULTS_PATH=
# Transformers
TRANSFORMERS_CACHE=
# Shard directory
SMALL_SHARD_DIR=
MEDIUM_SHARD_DIR=
SMALL_ENGLISH_FILTER_SHARD_DIR=
MEDIUM_ENGLISH_FILTER_SHARD_DIR=
# Annotated metadata file
SMALL_ENGLISH_FILTER_ANNOTATED_METADATA_DIR=
MEDIUM_ENGLISH_FILTER_ANNOTATED_METADATA_DIR=
# Domain Annotations
DOMAIN_ANNOTATION_CANDIDATE_DIR=
DOMAIN_ANNOTATION_OUTPUT_DIR=
# KEY
HUGGING_FACE_HUB_TOKEN=
LD_LIBRARY_PATH=
```
