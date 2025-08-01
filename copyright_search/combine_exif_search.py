import pandas as pd
import os
from dotenv import load_dotenv
import gc
from tqdm import tqdm
import logging
import argparse

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
        if os.path.isdir(os.path.join(result_dir, d))
    ]
    exif_match_df = pd.DataFrame()
    for shard_dir in tqdm(shard_dir_list):
        result_file_path = os.path.join(shard_dir, "metadata_with_exif_matches.parquet")
        df = pd.read_parquet(result_file_path, engine="pyarrow")
        sub_df = df.loc[df["exif_copyright_match"]]
        exif_match_df = pd.concat([exif_match_df, sub_df])
        del df
        del sub_df
        gc.collect()
    logger.info(f"Found {len(exif_match_df)} matches")
    exif_match_df.to_parquet(
        os.path.join(result_dir, "exif_match_subset.parquet"),
        engine="pyarrow",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    args = parser.parse_args()
    main(args)
