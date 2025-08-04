import pandas as pd
import argparse
import tldextract
from collections import Counter
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_base_domain(url):
    try:
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}"
    except:
        return None


def extract_full_domain(url):
    try:
        extracted = tldextract.extract(url)
        return f"{extracted.subdomain}.{extracted.domain}.{extracted.suffix}"
    except:
        return None


def main(args):
    # Read the metadata file
    input_path = Path(args.metadata_file)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_domain_output_path = output_dir / "base_domain_counts.json"
    full_domain_output_path = output_dir / "full_domain_counts.json"
    mapping_output_path = output_dir / "base_to_full_domain_mapping.json"
    df = pd.read_parquet(args.metadata_file)

    # Extract domains from URLs
    df["domains"] = df["url"].apply(extract_base_domain)
    df["full_domains"] = df["url"].apply(extract_full_domain)

    # Count domains
    domain_counts = Counter(df["domains"])
    full_domain_counts = Counter(df["full_domains"])
    # Create mapping of base domains to full domains
    # Group full domains by base domain and aggregate into lists of unique full domains
    base_to_full_domains = (
        df[["domains", "full_domains"]]
        .dropna()
        .groupby("domains")["full_domains"]
        .agg(lambda x: list(set(x)))
        .to_dict()
    )

    # Print top domains and their counts
    logger.info("Top 10 base domains by frequency:")
    for domain, count in domain_counts.most_common(10):
        logger.info(f"  {domain}: {count}")

    logger.info("Top 10 full domains by frequency:")
    for domain, count in full_domain_counts.most_common(10):
        logger.info(f"  {domain}: {count}")

    # Save full domain counts to file
    with open(base_domain_output_path, "w") as f:
        json.dump(dict(domain_counts), f, indent=4)
    with open(full_domain_output_path, "w") as f:
        json.dump(dict(full_domain_counts), f, indent=4)
    with open(mapping_output_path, "w") as f:
        json.dump(base_to_full_domains, f, indent=4)
    logger.info(f"Total unique domains: {len(domain_counts)}")
    logger.info(f"Base domain counts saved to {base_domain_output_path}")
    logger.info(f"Full domain counts saved to {full_domain_output_path}")
    logger.info(f"Base to full domain mapping saved to {mapping_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-file",
        type=str,
        required=True,
        help="Path to parquet metadata file containing URLs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory",
    )
    args = parser.parse_args()
    main(args)
