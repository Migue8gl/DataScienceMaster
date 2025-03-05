from collections import defaultdict
import logging
import os
import random
import time
from datetime import datetime
from typing import Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI


def setup_logger(log_level=logging.INFO):
    """Configures and returns a logger with file and stream handlers"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/hierarchical_data_{timestamp}.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def get_gpt_response(client: OpenAI, system_prompt: str, user_prompt: str, logger) -> str:
    """Get GPT response with error handling and rate limiting"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.05,
            max_tokens=1024
        )
        time.sleep(0.5)  # Basic rate limiting
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"API Error: {e}")
        time.sleep(2)
        raise


def create_hierarchical_categories(client: OpenAI, logger) -> Dict:
    """Create optimized hierarchical category structure with batched processing"""
    logger.info("Creating hierarchical category structure")

    # Generate main domains
    domains = get_gpt_response(
        client,
        system_prompt="You are an expert in scientific categorization.",
        user_prompt="Generate 8 distinct scientific domains that can be subdivided. Return comma-separated names.",
        logger=logger
    ).split(", ")[:8]  # Ensure exactly 5 domains

    hierarchy = {"categories": {}}

    for domain in domains:
        logger.info(f"Processing domain: {domain}")
        
        # Get subfields with validation
        subfields = get_gpt_response(
            client,
            system_prompt=f"You are a {domain} expert.",
            user_prompt=f"Generate 10-13 core subfields in {domain}. Return comma-separated names.",
            logger=logger
        ).split(", ")[:10]  # Limit to 7 subfields

        # Get specializations for all subfields in one batch
        specialization_prompt = (
            f"For each subfield in {domain}: {', '.join(subfields)}\n"
            "Generate 17-25 specific specializations per subfield. Format as:\n"
            "<subfield>: <comma-separated list>"
        )
        
        response = get_gpt_response(
            client,
            system_prompt=f"You are a {domain} expert with broad subfield knowledge.",
            user_prompt=specialization_prompt,
            logger=logger
        )

        # Parse specializations
        domain_specs = {}
        for line in response.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                subfield = parts[0].strip()
                if subfield in subfields:
                    specs = [s.strip() for s in parts[1].split(",") if s.strip()][:17]  # Limit to 12 specs
                    if specs:
                        domain_specs[subfield] = specs

        # Build domain structure with validation
        valid_subfields = {sf: {"specializations": specs} for sf, specs in domain_specs.items() if specs}
        if valid_subfields:
            hierarchy["categories"][domain] = {"subfields": valid_subfields}

    return hierarchy


def generate_hierarchical_text_data(n_samples: int = 2500, hierarchy: Dict = None) -> Tuple[pl.DataFrame, Dict]:
    """Generate text data with parallel API calls and efficient batching"""
    logger = setup_logger()
    logger.info(f"Starting text generation for {n_samples} samples")

    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not hierarchy:
        hierarchy = create_hierarchical_categories(client, logger)

    # Validate and flatten hierarchy
    valid_paths = []
    for domain, details in hierarchy["categories"].items():
        for subfield, specs in details["subfields"].items():
            if specs["specializations"]:
                valid_paths.extend([
                    (domain, subfield, spec) 
                    for spec in specs["specializations"]
                ])

    if not valid_paths:
        raise ValueError("No valid hierarchy paths found")

    # Distribute samples across paths
    path_counts = defaultdict(int)
    for _ in range(n_samples):
        path = random.choice(valid_paths)
        path_counts[path] += 1

    # Generate texts with parallel processing
    results = []
    BATCH_SIZE = 100  # Align with OpenAI API limits
    MAX_WORKERS = 1  # Control concurrent requests

    def process_batch(path, count):
        domain, subfield, spec = path
        system_prompt = f"Generate technical {spec} text in {domain}/{subfield} without mentioning categories."
        user_prompt = f"Write a detailed paragraph about a {spec} topic in {subfield}."

        texts = []
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                n=min(count, BATCH_SIZE),
                max_tokens=1024,
                temperature=0.05
            )
            texts = [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            logger.error(f"Batch failed for {path}: {e}")
        return [{"text": t, "domain": domain, "subfield": subfield, 
                "specialization": spec} for t in texts]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for path, count in path_counts.items():
            num_batches = (count + BATCH_SIZE - 1) // BATCH_SIZE
            for _ in range(num_batches):
                futures.append(executor.submit(process_batch, path, BATCH_SIZE))

        for future in as_completed(futures):
            results.extend(future.result())

    # Trim to exact sample count and shuffle
    random.shuffle(results)
    df = pl.DataFrame(results[:n_samples])

    return df, hierarchy

if __name__ == "__main__":
    try:
        logger = setup_logger()
        
        # Load or generate hierarchy
        hierarchy = None
        if os.path.exists("data/category_hierarchy.json"):
            import json
            with open("data/category_hierarchy.json", "r") as f:
                hierarchy = json.load(f)

        # Generate dataset
        df, hierarchy = generate_hierarchical_text_data(5500, hierarchy)

        # Save outputs
        df.write_csv("data/hierarchical_dataset.csv")
        with open("data/category_hierarchy.json", "w") as f:
            import json
            json.dump(hierarchy, f, indent=2)

        logger.info("Dataset generation completed successfully")

    except Exception as e:
        logger.error(f"Main execution failed: {e}", exc_info=True)
        raise