import pandas as pd
import os
from dotenv import load_dotenv
import gc
from tqdm import tqdm
import logging
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from regex_search.copyright_search import SEARCH_PATTERN_MAPPING

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    result_dir = args.base_dir
    shard_dir_list = [
        os.path.join(result_dir, d)
        for d in os.listdir(result_dir)
        if os.path.isdir(os.path.join(result_dir, d)) and "annotation" not in d
    ]
    result_df = pd.DataFrame()
    for shard_dir in tqdm(shard_dir_list):
        logger.info(
            f"Processing {os.path.join(shard_dir, args.copyright_input_filename)}"
        )
        copyright_match_df = pd.read_parquet(
            os.path.join(shard_dir, args.copyright_input_filename),
            engine="pyarrow",
        )
        result_df = pd.concat(
            [
                result_df,
                copyright_match_df.loc[copyright_match_df["copyright_match"]],
            ]
        )
        del copyright_match_df
        gc.collect()
    result_df.to_parquet(
        os.path.join(result_dir, "copyright_match_subset.parquet"), engine="pyarrow"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="The base directory of the shardified result",
    )
    parser.add_argument(
        "--copyright-input-filename",
        type=str,
        default="metadata_with_copyright_regex_matches.parquet",
    )
    args = parser.parse_args()
    main(args)
