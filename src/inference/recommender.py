# src/inference/recommender.py

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.preprocessing.graph_database import GraphDatabaseHandler
from src.preprocessing.image_processor import ImageProcessor
from src.preprocessing.vector_database import VectorDatabase


class Recommender:
    def __init__(
        self,
        graph_db: GraphDatabaseHandler,
        catalog_csv_path: str,
        vector_db: VectorDatabase,
    ):
        self.graph_db = graph_db
        # Load catalog data to get image paths and create mappings for quick lookup of product images and attributes
        self.catalog_df = pd.read_csv(catalog_csv_path)
        self.catalog_df["product_id"] = self.catalog_df["product_id"].astype('str')
        # Create mappings for quick lookup
        self.product_image_map = dict(
            zip(self.catalog_df["product_id"], self.catalog_df["image_path"])
        )
        self.product_attributes_map = self.catalog_df.set_index("product_id").to_dict(
            "index"
        )
        self.vector_db = vector_db
        self.processor = ImageProcessor(visualize_dir="temp_images")

    def get_recommendations(
        self,
        selected_product_id: str,
        filters: Dict[str, Any] = None,
        threshold: int = 1,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get recommended products for the selected_product_id.
        Returns a dictionary containing selected product info and recommendations.
        """
        if filters is None:
            filters = {}
        recommendations = self.graph_db.get_recommendations(
            selected_product_id, filters, threshold, top_k
        )
        # Get image paths and other info for the selected product
        selected_product_image = self.product_image_map.get(selected_product_id)

        # Prepare recommendations
        def process_recommendations(recs):
            processed = []
            for rec in recs:
                product_id = rec["product_id"]
                weight = rec["weight"]
                images = rec.get("images", [])
                metadata = rec["metadata"]  # Node properties
                # Convert metadata to dictionary
                metadata = dict(metadata)
                image_path = self.product_image_map.get(product_id)
                if image_path:
                    processed.append(
                        {
                            "product_id": product_id,
                            "image_path": image_path,
                            "weight": weight,
                            "images": images,
                            "attributes": metadata,
                        }
                    )
            return processed

        worn_with_recs = process_recommendations(recommendations["worn_with"])
        complemented_recs = process_recommendations(recommendations["complemented"])
        return {
            "selected_product": {
                "product_id": selected_product_id,
                "image_path": selected_product_image,
                "attributes": recommendations['selected_results'][0],
            },
            "worn_with": worn_with_recs,
            "complemented": complemented_recs,
        }

    def get_outfit_from_image(
        self,
        image_path_or_url: str,
        visualize: bool,
        image_id: str = "",
        similarity_threshold: float = 0.7,
        top_k: int = 1,
    ) -> Tuple[List[Dict[str, Any]], List[str | None]]:
        """
        Process the uploaded image, segment it into items, and find the closest matching
        products from the catalog for each item.

        Args:
            image_path (str): Path to the image to process or URL
            visualize (bool): Whether to visualize the segmented items
            image_id (str): Unique identifier for the image used to name the image
            similarity_threshold (float): Threshold for similarity matching. Only matches with
                scores >= this threshold will be included. Defaults to 0.7.
            top_k (int): Number of top matches to return per item. Defaults to 1.

        Returns:
            Tuple[List[Dict[str, Any]], List[str|None]]: A tuple containing:
                - List of matched products, where each product is a dictionary with:
                    - product_id (str): Unique identifier of the matched catalog product
                    - image_path (str): Path to the product image
                    - attributes (dict): Product metadata and attributes
                    - similarity_score (float): Matching similarity score
                    - item_type (str): Type of the item
                - List of file paths for segmented item images

        Notes:
            The function uses vector similarity search with filtering based on item type
            and gender. Items without a valid 'type' attribute will be skipped.
        """
        # Process the image
        items, filepaths = self.processor.process_image(
            image_path_or_url, visualize=visualize, image_id=image_id
        )

        matched_products = []
        for item in items:
            # Get embedding for the item
            embedding = item["embedding"].embedding
            # Get 'type' extracted by GPT-4O model
            item_type = item["attributes"].attributes.get("type")
            if not item_type:
                print(f"No 'type' found for item in image {image_path_or_url}")
                continue  # Skip this item if 'type' is missing
            # Set filters to retrieve items with the same 'type' and gender
            filters = {
                "type": item_type,
                "gender": {
                    "$in": ["unisex", item["attributes"].attributes.get("gender")]
                },
            }
            query_result = self.vector_db.query(
                embedding,
                top_k=top_k,
                namespace="catalog",
                include_values=False,
                filters=filters,
            )
            if query_result and "matches" in query_result and query_result["matches"]:
                for match in query_result["matches"]:
                    similarity_score = match["score"]
                    logger.debug(
                        f"Similarity with product_id: {match['id']} and score: {similarity_score} for item: {item}"
                    )
                    if similarity_score >= similarity_threshold:
                        catalog_product_id = match["id"]
                        logger.info(
                            f"Matching catalog item found: {catalog_product_id}"
                        )
                        metadata = match["metadata"]
                        product_info = {
                            "product_id": catalog_product_id,
                            "image_path": self.product_image_map.get(
                                catalog_product_id
                            ),
                            "attributes": metadata,
                            "similarity_score": similarity_score,
                            "item_type": item_type,
                        }
                        matched_products.append(product_info)
            else:
                print(
                    f"No matching catalog item found for item with type '{item_type}'"
                )
        return matched_products, filepaths


if __name__ == "__main__":
    rec = Recommender()
