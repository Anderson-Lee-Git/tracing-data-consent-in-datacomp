import pandas as pd
from dotenv import load_dotenv
import argparse
from pathlib import Path
import logging
import json

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def is_copyright_in_exif(exif_data):
    if exif_data:
        data = json.loads(exif_data)
        for key in data.keys():
            if "copyright" in key.lower():
                return True
            if "0x8298" in key.lower():
                return True
    return False


def extract_text_from_exif(exif_data):
    if exif_data:
        data = json.loads(exif_data)
        for key in data.keys():
            if "copyright" in key.lower():
                return data[key]
            if "0x8298" in key.lower():
                return data[key]
    return ""


def extract_key_from_exif(exif_data):
    if exif_data:
        data = json.loads(exif_data)
        for key in data.keys():
            if "copyright" in key.lower():
                return key
            if "0x8298" in key.lower():
                return key
    return ""


def main(args):
    logger.info(f"Loading metadata from {args.metadata_file}")
    df = pd.read_parquet(args.metadata_file, engine="pyarrow")
    df["exif_copyright_match"] = df["exif"].apply(is_copyright_in_exif)
    df["exif_copyright_text"] = df["exif"].apply(extract_text_from_exif)
    df["exif_copyright_key"] = df["exif"].apply(extract_key_from_exif)
    logger.info(f"Found {df['exif_copyright_match'].sum()} matches")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.metadata_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "metadata_with_exif_matches.parquet"
    logger.info(f"Writing to {output_file}")
    df.to_parquet(output_file, engine="pyarrow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="Path to metadata file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory",
    )
    args = parser.parse_args()
    main(args)
