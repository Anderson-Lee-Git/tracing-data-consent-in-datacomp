import argparse
import json
import os
import requests
import time
from pathlib import Path
from typing import Dict, List, Any
from urllib.parse import urljoin
import logging
from dotenv import load_dotenv
import shutil
from tqdm import tqdm

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def shardify_domains(
    base_to_full_domain_mapping: Dict[str, List[str]], num_shards: int, output_dir: Path
):
    full_domain_list = []
    for base_domain, full_domains in base_to_full_domain_mapping.items():
        full_domain_list.extend(full_domains)

    # create str shard id from 0 to num_shards - 1 with leading zeros
    shard_ids = [f"{i:04d}" for i in range(num_shards)]
    for i, shard_id in enumerate(tqdm(shard_ids)):
        sub_dir = output_dir / shard_id
        sub_dir.mkdir(parents=True, exist_ok=True)
        start_idx = i * len(full_domain_list) // num_shards
        end_idx = (i + 1) * len(full_domain_list) // num_shards
        if i == num_shards - 1:
            data = full_domain_list[start_idx:]
        else:
            data = full_domain_list[start_idx:end_idx]
        with open(sub_dir / "full_domains.json", "w") as f:
            json.dump(data, f, indent=4)


def load_shard_domains(output_dir: Path) -> List[str]:
    path = output_dir / "full_domains.json"
    if not path.exists():
        raise FileNotFoundError(f"Shard {output_dir} not found")
    with open(path, "r") as f:
        return json.load(f)


def load_existing_shard_results(output_path: Path) -> Dict[str, str]:
    path = output_path / "robots_txt.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_existing_results(output_path: Path) -> Dict[str, Dict[str, str]]:
    """Load existing results from checkpoint file if it exists."""
    if output_path.exists():
        logger.info(f"Loading existing results from: {output_path}")
        with open(output_path, "r") as f:
            return json.load(f)
    return {}


def load_top_domains(scale: str) -> List[str]:
    """Load top 50 domain list based on scale."""
    domain_annotation_dir = os.getenv("DOMAIN_ANNOTATION_CANDIDATE_DIR")
    if not domain_annotation_dir:
        raise ValueError("DOMAIN_ANNOTATION_CANDIDATE_DIR environment variable not set")

    file_path = Path(domain_annotation_dir) / f"top_50_en_{scale}_base_domains.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Domains file not found: {file_path}")

    with open(file_path, "r") as f:
        data = json.load(f)

    # Extract base domains from the annotations
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return list(data.keys())
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def load_domain_mapping(scale: str) -> Dict[str, List[str]]:
    """Load base to full domain mapping if scale is 'small'."""
    if scale == "small":
        metadata_dir = os.getenv("SMALL_ENGLISH_FILTER_ANNOTATED_METADATA_DIR", "")
    elif scale == "medium":
        metadata_dir = os.getenv("MEDIUM_ENGLISH_FILTER_ANNOTATED_METADATA_DIR", "")
    else:
        raise ValueError(f"Invalid scale: {scale}")

    file_path = Path(metadata_dir) / "base_to_full_domain_mapping.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Domain mapping file not found: {file_path}")

    with open(file_path, "r") as f:
        return json.load(f)


def fetch_robots_txt(domain: str, timeout: int = 30, max_retries: int = 1) -> str:
    """Fetch robots.txt content from a domain with retry logic for connection timeouts only."""
    # Ensure domain has proper protocol
    if not domain.startswith(("http://", "https://")):
        domain = f"https://{domain}"

    robots_url = urljoin(domain, "/robots.txt")
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RobotsTxtBot/1.0)"}

    # Try both HTTP and HTTPS if the first fails
    protocols_to_try = (
        ["https", "http"] if domain.startswith("https://") else ["http", "https"]
    )

    for attempt in range(max_retries):
        for protocol in protocols_to_try:
            try:
                # Construct URL with current protocol
                if protocol == "https":
                    test_url = robots_url.replace("http://", "https://")
                else:
                    test_url = robots_url.replace("https://", "http://")

                # Use longer timeout for slow servers
                response = requests.get(
                    test_url,
                    headers=headers,
                    timeout=(timeout, timeout),  # (connect_timeout, read_timeout)
                    allow_redirects=True,
                )

                if response.status_code == 200:
                    return response.text
                elif response.status_code in [301, 302, 307, 308]:
                    # Follow redirects manually if needed
                    continue
                else:
                    logger.warning(
                        f"Failed to fetch robots.txt from {test_url}: HTTP {response.status_code}"
                    )
                    return f"Error: HTTP {response.status_code}"

            except requests.exceptions.ConnectTimeout as e:
                logger.warning(
                    f"Connection timeout for {test_url} (attempt {attempt + 1}/{max_retries}): {str(e)}"
                )
                if attempt == max_retries - 1:
                    return f"Error: Connection timeout after {max_retries} attempts"
                # Only retry for connection timeouts
                continue

            except requests.exceptions.ReadTimeout as e:
                logger.warning(f"Read timeout for {test_url}: {str(e)}")
                return f"Error: Read timeout"

            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error for {test_url}: {str(e)}")
                return f"Error: Connection failed"

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed for {test_url}: {str(e)}")
                return f"Error: {str(e)}"

    return f"Error: Failed to fetch robots.txt after {max_retries} attempts"


def save_checkpoint(results: Dict[str, Dict[str, str]], output_path: Path):
    """Save current results as a checkpoint."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Checkpoint saved to: {output_path}")


def save_shard_checkpoint(results: Dict[str, str], output_path: Path):
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Checkpoint saved to: {output_path}")


def extract_robots_for_domains(
    base_domains: List[str],
    domain_mapping: Dict[str, List[str]],
    scale: str,
    output_path: Path,
    timeout: int = 30,
    max_retries: int = 3,
) -> Dict[str, Dict[str, str]]:
    """Extract robots.txt for all domains with checkpoint saving every 5000 full domains."""
    # Load existing results if checkpoint exists
    results = load_existing_results(output_path)

    # Count already processed full domains when resuming
    already_processed_full_domains = 0
    for base_domain, full_domain_results in results.items():
        already_processed_full_domains += len(full_domain_results)

    if already_processed_full_domains > 0:
        logger.info(
            f"Resuming: {already_processed_full_domains} full domains already processed"
        )

    full_domains_processed = already_processed_full_domains
    total_base_domains = len(base_domains)
    checkpoint_interval = 5000

    for i, base_domain in enumerate(base_domains, 1):
        logger.info(f"Processing base domain ({i}/{total_base_domains}): {base_domain}")

        # Initialize base domain results if not exists
        if base_domain not in results:
            results[base_domain] = {}

        if scale == "small" and base_domain in domain_mapping:
            # Use mapped full domains
            full_domains = domain_mapping[base_domain]
        elif scale == "medium" and base_domain in domain_mapping:
            # Use mapped full domains
            full_domains = domain_mapping[base_domain]
        else:
            # Use base domain itself
            full_domains = [base_domain]

        # Check if this base domain has been fully processed
        already_processed_for_base = len(results[base_domain])
        if already_processed_for_base == len(full_domains):
            logger.info(
                f"Skipping fully processed base domain ({i}/{total_base_domains}): {base_domain} ({already_processed_for_base}/{len(full_domains)} full domains)"
            )
            continue
        elif already_processed_for_base > 0:
            logger.info(
                f"Resuming partially processed base domain ({i}/{total_base_domains}): {base_domain} ({already_processed_for_base}/{len(full_domains)} full domains already processed)"
            )

        logger.info(f"Size of full domains: {len(full_domains)}")

        for j, full_domain in enumerate(full_domains, 1):
            # Skip if this full domain was already processed
            if (
                full_domain in results[base_domain]
                and "timeout" not in results[base_domain][full_domain]
            ):
                logger.info(
                    f"  Skipping already processed full domain ({j}/{len(full_domains)}): {full_domain}"
                )
                continue

            logger.info(
                f"  Fetching robots.txt for ({j}/{len(full_domains)}): {full_domain}"
            )
            robots_content = fetch_robots_txt(
                full_domain, timeout=timeout, max_retries=max_retries
            )
            results[base_domain][full_domain] = robots_content
            full_domains_processed += 1

            # Save checkpoint every 5000 full domains
            if full_domains_processed % checkpoint_interval == 0:
                save_checkpoint(results, output_path)
                logger.info(
                    f"Checkpoint saved after processing {full_domains_processed} full domains"
                )

            # Be respectful to servers
            time.sleep(1)

        logger.info(
            f"Completed base domain {base_domain} (total full domains processed: {full_domains_processed})"
        )

    # Save final checkpoint if there are remaining domains processed
    if full_domains_processed % checkpoint_interval != 0:
        save_checkpoint(results, output_path)
        logger.info(
            f"Final checkpoint saved with {full_domains_processed} total full domains processed"
        )

    return results


def extract_robots_for_shard(
    full_domains: List[str],
    scale: str,
    output_path: Path,
    shard_id: int,
    timeout: int = 30,
    max_retries: int = 3,
) -> Dict[str, str]:
    """Extract robots.txt for all domains with checkpoint saving every 5000 full domains.
    output_path is already a shard directory
    """
    # Load existing results if checkpoint exists

    results = load_existing_shard_results(output_path)
    domains_processed = 0
    checkpoint_interval = 200

    for i, domain in enumerate(full_domains):
        if domain in results:
            logger.info(
                f"Skipping already processed domain ({i}/{len(full_domains)}): {domain}"
            )
            continue

        logger.info(f"Fetching robots.txt for ({i}/{len(full_domains)}): {domain}")
        robots_content = fetch_robots_txt(
            domain, timeout=timeout, max_retries=max_retries
        )
        results[domain] = robots_content
        domains_processed += 1

        # Save checkpoint every 5000 domains
        if domains_processed % checkpoint_interval == 0:
            save_shard_checkpoint(results, output_path)
            logger.info(
                f"Checkpoint saved after processing {domains_processed} domains"
            )

        # Be respectful to servers
        time.sleep(1)

    # Save final checkpoint if there are remaining domains processed
    if domains_processed % checkpoint_interval != 0:
        save_shard_checkpoint(results, output_path)
        logger.info(
            f"Final checkpoint saved with {domains_processed} total domains processed"
        )

    return results


def save_results(
    results: Dict[str, Dict[str, str]], scale: str, output_dir: Path = Path(".")
):
    output_path = output_dir / f"robots_txt_{scale}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Results saved to: {output_path}")


def save_shard_results(
    results: Dict[str, str], scale: str, output_dir: Path = Path(".")
):
    output_path = output_dir / f"robots_txt_{scale}.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract robots.txt from web domains")
    parser.add_argument(
        "--mode",
        choices=["top", "full"],
        default="top",
        help="Mode to process domains (top or full)",
    )
    parser.add_argument(
        "--shard-id", type=int, help="Shard ID to process if mode is full"
    )
    parser.add_argument(
        "--scale",
        choices=["small", "medium"],
        required=True,
        help="Scale of domains to process (small or medium)",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to save output file (default: current directory)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout for HTTP requests in seconds (default: 30)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum number of retries for failed requests (default: 3)",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint file and start fresh",
    )

    args = parser.parse_args()

    try:
        if args.mode == "top":
            # TOP MODE FLOW
            logger.info("Running in TOP mode")

            # Setup output directory
            output_dir = Path(args.output_dir)

            # Handle checkpoint reset
            checkpoint_path = output_dir / f"robots_txt_{args.scale}_checkpoint.json"
            if args.reset_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted existing checkpoint: {checkpoint_path}")

            # Load top domains
            logger.info(f"Loading top domains for scale: {args.scale}")
            base_domains = load_top_domains(args.scale)
            logger.info(f"Loaded {len(base_domains)} base domains")

            # Load domain mapping
            logger.info("Loading domain mapping...")
            domain_mapping = load_domain_mapping(args.scale)
            if domain_mapping:
                total_mapped_domains = sum(
                    len(domains) for domains in domain_mapping.values()
                )
                logger.info(
                    f"Loaded domain mapping with {len(domain_mapping)} base domains mapping to {total_mapped_domains} full domains"
                )

            # Setup output paths
            output_path = output_dir / f"robots_txt_{args.scale}_checkpoint.json"
            final_output_path = output_dir / f"robots_txt_{args.scale}.json"

            # Show resume information if checkpoint exists
            if output_path.exists():
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
                logger.info(
                    f"Found existing checkpoint with {len(existing_data)} base domains already processed"
                )
                logger.info("Resuming from checkpoint...")

            # Extract robots.txt
            total_number_of_domains_to_process = 0
            for base_domain, full_domains in domain_mapping.items():
                if base_domain in base_domains:
                    total_number_of_domains_to_process += len(full_domains)
            logger.info(
                f"Total number of domains to process: {total_number_of_domains_to_process}"
            )
            logger.info("Starting robots.txt extraction...")
            results = extract_robots_for_domains(
                base_domains,
                domain_mapping,
                args.scale,
                output_path,
                args.timeout,
                args.max_retries,
            )

            # Save final results (rename checkpoint to final file)
            logger.info("All domains processed successfully!")
            if output_path.exists():
                # Move checkpoint to final output file
                shutil.move(str(output_path), str(final_output_path))
                logger.info(f"Final results saved to: {final_output_path}")
            else:
                # Fallback: save results normally
                save_results(results, args.scale, output_dir)

            # Print summary
            total_domains = sum(len(sub_dict) for sub_dict in results.values())
            logger.info(
                f"Extraction complete! Processed {total_domains} domains across {len(results)} base domains"
            )

        elif args.mode == "full":
            # FULL MODE FLOW
            logger.info("Running in FULL mode")

            # Validate shard ID
            assert args.shard_id is not None, "Shard ID is required for full mode"
            shard_id_str = f"{args.shard_id:04d}"

            # Setup output directory
            output_dir = Path(args.output_dir) / shard_id_str
            output_dir.mkdir(parents=True, exist_ok=True)

            # Handle checkpoint reset
            checkpoint_path = output_dir / f"robots_txt_{args.scale}_checkpoint.json"
            if args.reset_checkpoint and checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted existing checkpoint: {checkpoint_path}")

            # Load shard domains
            logger.info(f"Loading shard {shard_id_str}'s domains...")
            full_domains = load_shard_domains(output_dir)
            logger.info(
                f"Loaded {len(full_domains)} full domains for shard {shard_id_str}"
            )

            # Setup output paths
            output_path = output_dir / f"robots_txt_{args.scale}_checkpoint.json"
            final_output_path = output_dir / f"robots_txt_{args.scale}.json"

            # Show resume information if checkpoint exists
            if output_path.exists():
                with open(output_path, "r") as f:
                    existing_data = json.load(f)
                logger.info(
                    f"Found existing checkpoint with {len(existing_data)} domains already processed"
                )
                logger.info("Resuming from checkpoint...")

            # Extract robots.txt
            logger.info("Starting robots.txt extraction...")
            results = extract_robots_for_shard(
                full_domains,
                args.scale,
                output_path,
                args.shard_id,
                args.timeout,
                args.max_retries,
            )

            # Save final results (rename checkpoint to final file)
            logger.info("All domains processed successfully!")
            if output_path.exists():
                # Move checkpoint to final output file
                shutil.move(str(output_path), str(final_output_path))
                logger.info(f"Final results saved to: {final_output_path}")
            else:
                # Fallback: save results normally
                save_shard_results(results, args.scale, output_dir)

            # Print summary
            total_domains = len(results)
            logger.info(
                f"Extraction complete! Processed {total_domains} domains in shard {shard_id_str}"
            )
            logger.info(
                f"Success rate: {sum(1 for v in results.values() if not v.startswith('Error'))}/{total_domains} ({sum(1 for v in results.values() if not v.startswith('Error'))/total_domains*100:.2f}%)"
            )

        else:
            raise ValueError(f"Invalid mode: {args.mode}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
