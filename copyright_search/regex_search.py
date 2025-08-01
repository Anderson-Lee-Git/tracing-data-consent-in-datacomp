import pandas as pd
from dotenv import load_dotenv
import argparse
from pathlib import Path
import re
import logging

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


SEARCH_PATTERN_MAPPING = {
    "rights-general": [
        r"rights\s+secured",
        r"rights\s+reserved",
        r"licensed\s+by",
        r"under\s+license",
        r"owned\s+by",
    ],
    "copyright-general": [
        r"copyright(?:ed|s)?",
        r"\(c\)",
        r"copr\.?",
    ],
    "copyright-symbol": [
        r"[©Ⓒⓒ]",
    ],
    "trademark-symbol": [
        r"[®]",
    ],
    "cc-by": [
        r"cc licenses",
        r"cc 4\.0",
        r"cc 3\.0",
        r"cc 2\.5",
        r"cc 2\.0",
        r"cc 1\.0",
        r"cc by",
        r"cc by 4\.0",
        r"cc by 3\.0",
        r"cc by 2\.5",
        r"cc by 2\.0",
        r"cc by 1\.0",
    ],
    "cc-by-sa": [
        r"cc by-sa",
        r"cc by-sa 4\.0",
        r"cc by-sa 3\.0",
        r"cc by-sa 2\.5",
        r"cc by-sa 2\.0",
        r"cc by-sa 1\.0",
    ],
    "cc-by-nc": [
        r"cc by-nc",
        r"cc by-nc 4\.0",
        r"cc by-nc 3\.0",
        r"cc by-nc 2\.5",
        r"cc by-nc 2\.0",
        r"cc by-nc 1\.0",
    ],
    "cc-by-nc-sa": [
        r"cc by-nc-sa",
        r"cc by-nc-sa 4\.0",
        r"cc by-nc-sa 3\.0",
        r"cc by-nc-sa 2\.5",
        r"cc by-nc-sa 2\.0",
        r"cc by-nc-sa 1\.0",
    ],
    "cc-by-nc-nd": [
        r"cc by-nc-nd",
        r"cc by-nc-nd 4\.0",
        r"cc by-nc-nd 3\.0",
        r"cc by-nc-nd 2\.5",
        r"cc by-nc-nd 2\.0",
        r"cc by-nc-nd 1\.0",
    ],
    "cc-by-nd": [
        r"cc by-nd",
        r"cc by-nd 4\.0",
        r"cc by-nd 3\.0",
        r"cc by-nd 2\.5",
        r"cc by-nd 2\.0",
        r"cc by-nd 1\.0",
    ],
    "cc-zero": [
        r"cc0",
        r"cc0 1\.0",
    ],
    "no-restriction": [
        r"no\s+restriction",
        r"no\s+copyright",
    ],
}


def build_pattern(pattern_key):
    search_words = SEARCH_PATTERN_MAPPING[pattern_key]
    if pattern_key in ["copyright-symbol", "trademark-symbol"]:
        # Special handling for symbols
        pattern = (
            r"\s*(" + "|".join(search_words) + r")\s*(?:[^\n]*?\d{4}[^\n]*?|[^\n]*)"
        )
    else:
        # Special handling for copyright-general with (c)
        patterns = []
        for word in search_words:
            if word == r"\(c\)":
                patterns.append(word)
            else:
                patterns.append(r"\b" + word + r"\b")
        pattern = "(" + "|".join(patterns) + ")"
    return pattern


def main(args):
    logger.info(f"Loading metadata from {args.metadata_file}")
    df = pd.read_parquet(args.metadata_file, engine="pyarrow")
    if df["ocr_text"].isna().sum() > 0:
        raise ValueError(
            f"Found {df['ocr_text'].isna().sum()} ocr_text with null values"
        )
    for search_pattern in SEARCH_PATTERN_MAPPING.keys():
        copyright_search_pattern = build_pattern(search_pattern)
        ignore_search_words = [r"lorem", r"ipsum"]
        ignore_search_pattern = r"\b(" + "|".join(ignore_search_words) + r")\b"
        df[f"{search_pattern}_caption_match"] = df["caption"].str.contains(
            copyright_search_pattern, regex=True, case=False
        ) & ~df["caption"].str.contains(ignore_search_pattern, regex=True, case=False)
        df[f"{search_pattern}_ocr_text_match"] = df["ocr_text"].str.contains(
            copyright_search_pattern, regex=True, case=False
        ) & ~df["ocr_text"].str.contains(ignore_search_pattern, regex=True, case=False)
        df[f"{search_pattern}_match"] = (
            df[f"{search_pattern}_caption_match"]
            | df[f"{search_pattern}_ocr_text_match"]
        )
        logger.info(
            f"Found {df[f'{search_pattern}_caption_match'].sum()} caption matches for {search_pattern}"
        )
        logger.info(
            f"Found {df[f'{search_pattern}_ocr_text_match'].sum()} ocr_text matches for {search_pattern}"
        )
    count_as_match = [
        "rights-general",
        "copyright-general",
        "copyright-symbol",
        "cc-by",
        "cc-by-sa",
        "cc-by-nc",
        "cc-by-nc-sa",
        "cc-by-nc-nd",
        "cc-by-nd",
    ]
    caption_count_as_match = [f"{c}_caption_match" for c in count_as_match]
    ocr_text_count_as_match = [f"{c}_ocr_text_match" for c in count_as_match]
    general_count_as_match = [f"{c}_match" for c in count_as_match]
    df["copyright_caption_match"] = df[caption_count_as_match].sum(axis=1) >= 1
    df["copyright_ocr_text_match"] = df[ocr_text_count_as_match].sum(axis=1) >= 1
    df["copyright_match"] = df[general_count_as_match].sum(axis=1) >= 1
    output_file = (
        Path(args.metadata_file).parent
        / "metadata_with_copyright_regex_matches.parquet"
    )
    logger.info(f"Writing to {output_file}")
    df.to_parquet(output_file, engine="pyarrow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-file", type=str, required=True)
    args = parser.parse_args()
    main(args)
