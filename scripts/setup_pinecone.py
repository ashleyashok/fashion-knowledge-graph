#!/usr/bin/env python3
"""
Setup script for Pinecone vector database indexes.
This script creates the necessary indexes for the Complete the Look application.
"""

import os
import sys
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import pinecone
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def setup_pinecone_indexes() -> None:
    """Set up Pinecone indexes for the application."""

    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        sys.exit(1)

    pinecone.init(api_key=api_key)

    # Index configurations
    indexes_config = {
        "catalog-clothes": {
            "dimension": 512,  # CLIP embedding dimension
            "metric": "cosine",
            "pod_type": "p1.x1",  # Adjust based on your needs
        },
        "catalog-style-description": {
            "dimension": 384,  # Sentence transformer dimension
            "metric": "cosine",
            "pod_type": "p1.x1",
        },
    }

    # Create indexes
    for index_name, config in indexes_config.items():
        try:
            # Check if index already exists
            if index_name in pinecone.list_indexes():
                logger.info(f"Index '{index_name}' already exists")
                continue

            # Create new index
            logger.info(f"Creating index '{index_name}' with config: {config}")
            pinecone.create_index(
                name=index_name,
                dimension=config["dimension"],
                metric=config["metric"],
                pod_type=config["pod_type"],
            )
            logger.success(f"Successfully created index '{index_name}'")

        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            continue

    # List all indexes
    logger.info("Current Pinecone indexes:")
    for index_name in pinecone.list_indexes():
        logger.info(f"  - {index_name}")


def get_index_stats() -> Dict[str, Any]:
    """Get statistics for all indexes."""

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        logger.error("PINECONE_API_KEY not found in environment variables")
        return {}

    pinecone.init(api_key=api_key)

    stats = {}
    for index_name in pinecone.list_indexes():
        try:
            index = pinecone.Index(index_name)
            stats[index_name] = index.describe_index_stats()
            logger.info(f"Index '{index_name}' stats: {stats[index_name]}")
        except Exception as e:
            logger.error(f"Error getting stats for index '{index_name}': {e}")

    return stats


def main():
    """Main function to set up Pinecone indexes."""

    logger.info("Setting up Pinecone indexes for Complete the Look...")

    # Set up indexes
    setup_pinecone_indexes()

    # Get and display stats
    logger.info("Getting index statistics...")
    get_index_stats()

    logger.success("Pinecone setup completed successfully!")


if __name__ == "__main__":
    main()
