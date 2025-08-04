import json
import logging
import os
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(args):
    # Get all shard directories
    base_dir = Path(args.base_dir)
    shard_dirs = [
        d
        for d in base_dir.iterdir()
        if d.is_dir()
        and "annotation" not in d.name
        and "robots_txt" not in d.name
    ]

    # Initialize counter for all domains
    total_base_domain_counts = Counter()
    total_full_domain_counts = Counter()
    base_to_full_domains = {}

    # Aggregate counts from each shard
    for shard_dir in tqdm(shard_dirs):
        domain_counts_file = shard_dir / "base_domain_counts.json"
        if domain_counts_file.exists():
            with open(domain_counts_file, "r") as f:
                shard_counts = json.load(f)
                total_base_domain_counts.update(shard_counts)
        domain_counts_file = shard_dir / "full_domain_counts.json"
        if domain_counts_file.exists():
            with open(domain_counts_file, "r") as f:
                shard_counts = json.load(f)
                total_full_domain_counts.update(shard_counts)
        mapping_file = shard_dir / "base_to_full_domain_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, "r") as f:
                shard_mapping = json.load(f)
                for base_domain, full_domains in shard_mapping.items():
                    if base_domain not in base_to_full_domains:
                        base_to_full_domains[base_domain] = set()
                    base_to_full_domains[base_domain].update(full_domains)

    # Print top domains and their counts
    logger.info("Top 10 base domains across all shards:")
    for domain, count in total_base_domain_counts.most_common(10):
        logger.info(f"  {domain}: {count}")
    logger.info("Top 10 full domains across all shards:")
    for domain, count in total_full_domain_counts.most_common(10):
        logger.info(f"  {domain}: {count}")
    logger.info("Top base domains with most number of different full domains:")
    base_to_full_domains_list = sorted(
        list(base_to_full_domains.items()), key=lambda x: len(x[1]), reverse=True
    )
    for base_domain, full_domains in base_to_full_domains_list[:10]:
        logger.info(f"  {base_domain}: {len(full_domains)}")

    # Save aggregated counts
    output_path = base_dir / "base_domain_counts.json"
    with open(output_path, "w") as f:
        json.dump(dict(total_base_domain_counts), f, indent=4)
    output_path = base_dir / "full_domain_counts.json"
    with open(output_path, "w") as f:
        json.dump(dict(total_full_domain_counts), f, indent=4)
    output_path = base_dir / "base_to_full_domain_mapping.json"
    with open(output_path, "w") as f:
        base_to_full_domains = {k: list(v) for k, v in base_to_full_domains.items()}
        json.dump(base_to_full_domains, f, indent=4)

    logger.info(
        f"Total unique base domains across all shards: {len(total_base_domain_counts)}"
    )
    logger.info(
        f"Total unique full domains across all shards: {len(total_full_domain_counts)}"
    )

    with open(
        Path(os.getenv("DOMAIN_ANNOTATION_CANDIDATE_DIR"))
        / args.output_annotation_candidate_filename,
        "w",
    ) as f:
        json.dump(
            [x[0] for x in total_base_domain_counts.most_common(args.top_k)],
            f,
            indent=4,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True)
    # top k to save
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument(
        "--output-annotation-candidate-filename",
        "-o",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
