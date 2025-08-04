# Copied and modified from https://github.com/Data-Provenance-Initiative/Data-Provenance-Collection/blob/main/src/web_analysis/parse_robots.py

import argparse
import gzip
import json
import gc
import psutil
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd


def parse_robots_txt(robots_txt):
    """Parses the robots.txt to create a map of agents, sitemaps, and parsing oddities."""
    rules = {"ERRORS": [], "Sitemaps": []}
    agent_map = {}
    current_agents = []
    seen_agent_criteria = False

    for raw_line in robots_txt.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        lower_line = line.lower()

        if lower_line.startswith("user-agent:"):
            agent_name_raw = line.split(":", 1)[1].strip()
            agent_name = agent_name_raw.lower()

            if agent_name not in agent_map:
                agent_map[agent_name] = agent_name_raw
                rules[agent_name_raw] = defaultdict(list)  # type: ignore
            if seen_agent_criteria:
                current_agents = [agent_name]
            else:
                current_agents.append(agent_name)

        elif any(lower_line.startswith(d + ":") for d in ["allow", "disallow"]):
            seen_agent_criteria = True
            if not current_agents:
                rules["ERRORS"].append(
                    f"Directive '{line}' found with no preceding user-agent."
                )
                continue
            directive_name = lower_line.split(":", 1)[0].strip()
            directive_value = line.split(":", 1)[1].strip()
            directive_key = directive_name.capitalize()
            for agent in current_agents:
                original_name = agent_map[agent]
                rules[original_name][directive_key].append(directive_value)
        elif lower_line.startswith("crawl-delay:"):
            seen_agent_criteria = True
            crawl_delay_value = line.split(":", 1)[1].strip()
            for agent in current_agents:
                original_name = agent_map[agent]
                rules[original_name]["Crawl-Delay"].append(crawl_delay_value)  # type: ignore
        elif lower_line.startswith("sitemap:"):
            seen_agent_criteria = True
            sitemap_value = line.split(":", 1)[1].strip()
            rules["Sitemaps"].append(sitemap_value)
        else:
            rules["ERRORS"].append(f"Unmatched line: {raw_line}")

    return rules


def parallel_parse_robots(data, chunk_size=10000):
    """Takes in {URL --> robots text}

    Returns {URL --> {ERRORS/Sitemap/Agent --> {Allow/Disallow --> [paths...]}}
    """
    # populate the empty rules (missing robots.txt)
    url_to_rules = {url: {} for url, txt in data.items() if not txt}

    # Process in chunks to reduce memory usage
    data_items = [(url, txt) for url, txt in data.items() if txt]

    for i in range(0, len(data_items), chunk_size):
        chunk = data_items[i : i + chunk_size]

        # interpret the robots.txt
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_url = {
                executor.submit(parse_robots_txt, txt): url for url, txt in chunk
            }
            for future in tqdm(
                as_completed(future_to_url),
                total=len(future_to_url),
                desc=f"Parsing robots.txt (chunk {i//chunk_size + 1}/{(len(data_items) + chunk_size - 1)//chunk_size})",
            ):
                url = future_to_url[future]
                try:
                    url_to_rules[url] = future.result()
                except Exception as e:
                    print(f"Error processing {url}: {e}")

    print(f"Parsed {len(url_to_rules)} robots.txt")
    return url_to_rules


def interpret_agent(rules):

    agent_disallow = [x for x in rules.get("Disallow", []) if "?" not in x]
    agent_allow = [x for x in rules.get("Allow", []) if "?" not in x]

    if (
        len(agent_disallow) == 0
        or agent_disallow == [""]
        or (agent_allow == agent_disallow)
    ):
        disallow_type = "none"
    elif any("/" == x.strip() for x in agent_disallow) and len(agent_allow) == 0:
        disallow_type = "all"
    else:
        disallow_type = "some"

    return disallow_type


def interpret_robots(agent_rules, all_agents):
    """Given the robots.txt agent rules, and a list of relevant agents
    (a superset of the robots.txt), determine whether they are:

    (1) blocked from scraping all parts of the website
    (2) blocked from scraping part of the website
    (3) NOT blocked from scraping at all
    """
    # agent --> "all", "some", or "none" blocked.
    agent_to_judgement = {}

    star_judgement = interpret_agent(agent_rules.get("*", {}))
    agent_to_judgement["*"] = star_judgement

    for agent in all_agents:
        rule = agent_rules.get(agent)
        judgement = (
            interpret_agent(rule) if rule is not None else agent_to_judgement["*"]
        )
        agent_to_judgement[agent] = judgement

    return agent_to_judgement


def aggregate_robots(url_to_rules, all_agents, full_domain_counts, chunk_size=10000):
    """Across all robots.txt, determine basic stats:
    (1) For each agent, how often it is explicitly mentioned
    (2) For each agent, how often it is blocked from scraping all parts of the website
    (3) For each agent, how often it is blocked from scraping part of the website
    (4) For each agent, how often it is NOT blocked from scraping at all
    """
    robots_stats = defaultdict(lambda: {"counter": 0, "all": 0, "some": 0, "none": 0})

    # Process URLs in chunks to reduce memory usage
    url_items = list(url_to_rules.items())

    for i in range(0, len(url_items), chunk_size):
        chunk = url_items[i : i + chunk_size]

        for url, robots in tqdm(
            chunk,
            total=len(chunk),
            desc=f"Aggregating robots.txt (chunk {i//chunk_size + 1}/{(len(url_items) + chunk_size - 1)//chunk_size})",
        ):
            sample_counts = full_domain_counts[url] if url in full_domain_counts else 1
            agent_to_judgement = interpret_robots(robots, all_agents)

            # Trace individual agents and wildcard agent
            for agent in all_agents + ["*"]:
                if agent in robots:  # Missing robots.txt means nothing blocked
                    robots_stats[agent]["counter"] += sample_counts
                    judgement = agent_to_judgement[agent]
                    robots_stats[agent][judgement] += sample_counts

            # See if *All Agents* are blocked on all content,
            # or at least some agents can access some or more content, or
            # there are no blocks on any agents at all.
            if all(v == "all" for v in agent_to_judgement.values()):
                agg_judgement = "all"
            elif any(v in ["some", "all"] for v in agent_to_judgement.values()):
                agg_judgement = "some"
            else:
                agg_judgement = "none"
            robots_stats["*All Agents*"]["counter"] += sample_counts
            robots_stats["*All Agents*"][agg_judgement] += sample_counts

    return robots_stats


def aggregate_robots_with_url_decisions(
    url_to_rules, all_agents, full_domain_counts, chunk_size=10000
):
    """Memory-optimized version that also returns url_decisions if needed"""
    robots_stats = defaultdict(lambda: {"counter": 0, "all": 0, "some": 0, "none": 0})
    url_decisions = {}

    # Process URLs in chunks to reduce memory usage
    url_items = list(url_to_rules.items())

    for i in range(0, len(url_items), chunk_size):
        chunk = url_items[i : i + chunk_size]

        for url, robots in tqdm(
            chunk,
            total=len(chunk),
            desc=f"Aggregating robots.txt (chunk {i//chunk_size + 1}/{(len(url_items) + chunk_size - 1)//chunk_size})",
        ):
            sample_counts = full_domain_counts[url] if url in full_domain_counts else 1
            agent_to_judgement = interpret_robots(robots, all_agents)
            url_decisions[url] = agent_to_judgement

            # Trace individual agents and wildcard agent
            for agent in all_agents + ["*"]:
                if agent in robots:  # Missing robots.txt means nothing blocked
                    robots_stats[agent]["counter"] += sample_counts
                    judgement = agent_to_judgement[agent]
                    robots_stats[agent][judgement] += sample_counts

            # See if *All Agents* are blocked on all content,
            # or at least some agents can access some or more content, or
            # there are no blocks on any agents at all.
            if all(v == "all" for v in agent_to_judgement.values()):
                agg_judgement = "all"
            elif any(v in ["some", "all"] for v in agent_to_judgement.values()):
                agg_judgement = "some"
            else:
                agg_judgement = "none"
            robots_stats["*All Agents*"]["counter"] += sample_counts
            robots_stats["*All Agents*"][agg_judgement] += sample_counts
            url_decisions[url]["*All Agents*"] = agg_judgement

    return robots_stats, url_decisions


def analyze_robots(
    data, full_domain_counts={}, return_url_interpretations=False, chunk_size=10000
):
    """Takes in {URL --> robots text}

    Returns:
        robots_stats: {Agent --> {count: <int>, all: <int>, some: <int>, none: <int>}}
            where `count` are number of observations of this agent
            `all` is how many times its had all paths blocked
            `some` is how many times its had some paths blocked
            `none` is how many times its had none paths blocked
        url_interpretations: {URL --> {Agent --> <all/some/none>}} (only if return_url_interpretations=True)
            Only includes agents seen at this URL.
            Agents not seen at this URL will inherit "*" rules, or `none` at all.
    """
    url_to_rules = parallel_parse_robots(data, chunk_size)
    # Collect all agents
    all_agents = list(
        set(
            [
                agent
                for url, rules in url_to_rules.items()
                for agent, _ in rules.items()
                if agent not in ["ERRORS", "Sitemaps", "*"]
            ]
        )
    )

    # del data to save memory
    del data
    gc.collect()

    if return_url_interpretations:
        robots_stats, url_decisions = aggregate_robots_with_url_decisions(
            url_to_rules, all_agents, full_domain_counts, chunk_size
        )

        # URL --> agents in its robots.txt and their decisions (all/some/none).
        url_interpretations = {
            k: {
                agent: v
                for agent, v in vs.items()
                if agent not in ["ERRORS", "Sitemaps"]
                and agent in list(url_to_rules[k]) + ["*All Agents*"]
            }
            for k, vs in tqdm(
                url_decisions.items(),
                total=len(url_decisions),
                desc="Generate url interpretations",
            )
        }

        # Clean up intermediate data
        del url_decisions
        del url_to_rules
        gc.collect()

        return robots_stats, url_interpretations
    else:
        robots_stats = aggregate_robots(
            url_to_rules, all_agents, full_domain_counts, chunk_size
        )

        # Clean up intermediate data
        del url_to_rules
        gc.collect()

        return robots_stats, None


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def read_robots_file(file_path, chunk_size=10000):
    """Memory-optimized file reading that processes data in chunks"""
    if file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as file:
            raw_data = json.load(file)
    elif file_path.endswith(".json"):
        with open(file_path, "r") as file:
            raw_data = json.load(file)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    first_value_item = next(iter(raw_data.values()))
    if not isinstance(first_value_item, str):
        # flatten the data in chunks to reduce memory usage
        data = {}
        raw_items = list(raw_data.items())

        for i in range(0, len(raw_items), chunk_size):
            chunk = raw_items[i : i + chunk_size]
            for _, v in tqdm(
                chunk,
                total=len(chunk),
                desc=f"Flattening data (chunk {i//chunk_size + 1}/{(len(raw_items) + chunk_size - 1)//chunk_size})",
            ):
                for k, v1 in v.items():
                    data[k] = v1
        # Clean up raw_data to free memory
        del raw_data
    else:
        data = raw_data
    return data


def main(args):
    # for each url counts, weighted by the sample counts
    print(f"Loading full domain counts from {args.full_domain_counts_path}")
    full_domain_counts = json.load(open(args.full_domain_counts_path))
    print(f"Memory usage after loading domain counts: {get_memory_usage():.1f} MB")

    print(f"Loading robots.txt from {args.file_path}")
    data = read_robots_file(args.file_path, chunk_size=args.chunk_size)
    print(f"Memory usage after loading robots.txt: {get_memory_usage():.1f} MB")
    num_urls = len(data)
    print(f"Total URLs: {num_urls}")
    # filter out http errors
    data = {k: v for k, v in data.items() if not v.startswith("Error: ") and v}
    num_samples = sum([cnt for url, cnt in full_domain_counts.items() if url in data])
    print(f"URLs with robots.txt: {len(data)}")
    print(f"Total number of samples with robots.txt: {num_samples}")

    robots_stats, url_interpretations = analyze_robots(
        data,
        full_domain_counts,
        return_url_interpretations=False,
        chunk_size=args.chunk_size,
    )
    print(f"Memory usage after analysis: {get_memory_usage():.1f} MB")

    # PRINT OUT INFO ON INDIVIDUAL URLS:
    # for url, interp in url_interpretations.items():
    #     if interp.get("*") == "all":
    #         print(url)
    # import pdb; pdb.set_trace()
    # print(url_interpretations["http://www.machinenoveltranslation.com/robots.txt"])

    sorted_robots_stats = sorted(
        robots_stats.items(),
        key=lambda x: x[1]["counter"] / num_urls if num_urls > 0 else 0,
        reverse=True,
    )

    print(
        f"{'Agent':<30} {'Observed': <10} {'Blocked All': >15} {'Blocked Some': >15} {'Blocked None': >15}"
    )
    for i, (agent, stats) in enumerate(sorted_robots_stats):
        data_size = stats["counter"]
        counter_pct = stats["counter"] / data_size * 100 if data_size > 0 else 0
        all_pct = stats["all"] / data_size * 100 if data_size > 0 else 0
        some_pct = stats["some"] / data_size * 100 if data_size > 0 else 0
        none_pct = stats["none"] / data_size * 100 if data_size > 0 else 0

        global_counter_pct = (
            stats["counter"] / num_samples * 100 if num_samples > 0 else 0
        )
        global_all_pct = stats["all"] / num_samples * 100 if num_samples > 0 else 0
        global_some_pct = stats["some"] / num_samples * 100 if num_samples > 0 else 0
        global_none_pct = stats["none"] / num_samples * 100 if num_samples > 0 else 0

        print_outs = [
            f"{agent:<20}",
            f"{stats['counter']:>5} ({counter_pct:5.1f}%) ({global_counter_pct:5.1f}%) {'':>5} ",
            f"{stats['all']:>5} ({all_pct:5.1f}%) ({global_all_pct:5.1f}%) {'':>5} ",
            f"{stats['some']:>5} ({some_pct:5.1f}%) ({global_some_pct:5.1f}%) {'':>5} ",
            f"{stats['none']:>5} ({none_pct:5.1f}%) ({global_none_pct:5.1f}%)",
        ]
        print(" ".join(print_outs))
        if i > 15:
            print(f"........")
            break

    # Save results to CSV
    results = []
    for agent, stats in sorted_robots_stats:
        results.append(
            {
                "agent": agent,
                "observed": stats["counter"],
                "blocked all": stats["all"],
                "blocked some": stats["some"],
                "blocked none": stats["none"],
            }
        )

    df = pd.DataFrame(results)
    if args.output_path:
        df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    """
    Example commands:

    python src/web_analysis/parse_robots.py <in-path> <out-path>

    python src/web_analysis/parse_robots.py data/robots-test.json.gz data/robots-analysis.csv

    Process:
        1. Reads the json.gz mapping base-url to robots.txt
        2. Parse all the robots.txt so they can be analyzed on aggregate
    """
    parser = argparse.ArgumentParser(description="Parse and analyze robots.txt.")
    parser.add_argument(
        "--file-path",
        type=str,
        help="Path to the JSON.GZ file mapping urls to robots.txt text.",
    )
    parser.add_argument(
        "--full-domain-counts-path",
        type=str,
        help="Path to the JSON file mapping full domains to counts.",
    )
    parser.add_argument(
        "--output-path", type=str, help="Path where analysis will be saved."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for memory optimization (default: 10000)",
    )
    args = parser.parse_args()
    main(args)
