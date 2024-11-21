# src/inference/recommender.py

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from src.database.graph_database import GraphDatabaseHandler
from src.database.vector_database import VectorDatabase
from src.models.model_manager import image_processor, embedding_model


class Recommender:
    def __init__(
        self,
        graph_db: GraphDatabaseHandler,
        catalog_csv_path: str,
        vector_db: VectorDatabase,
    ):
        """
        Initialize the Recommender system.

        Parameters
        ----------
        graph_db : GraphDatabaseHandler
            An instance of GraphDatabaseHandler for interacting with the graph database.
        catalog_csv_path : str
            Path to the catalog CSV file containing product data.
        vector_db : VectorDatabase
            An instance of VectorDatabase for vector similarity queries.
        """
        self.graph_db = graph_db
        # Load catalog data to get image paths and attributes
        self.catalog_df = pd.read_csv(catalog_csv_path)
        self.catalog_df["product_id"] = self.catalog_df["product_id"].astype(str)
        # Create mappings for quick lookup
        self.product_image_map = dict(
            zip(self.catalog_df["product_id"], self.catalog_df["image_path"])
        )
        self.product_attributes_map = self.catalog_df.set_index("product_id").to_dict(
            "index"
        )
        self.vector_db = vector_db
        self.processor = image_processor

    def get_recommendations(
        self,
        selected_product_id: str,
        filters: Optional[Dict[str, Any]] = None,
        threshold: int = 1,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get recommended products for a selected product.

        Parameters
        ----------
        selected_product_id : str
            The product ID of the selected product.
        filters : dict, optional
            A dictionary of filters to apply when retrieving recommendations.
            For example, {'type': 'shirt'}. Default is None.
        threshold : int, optional
            The minimum weight threshold for relationships to consider. Default is 1.
        top_k : int, optional
            The maximum number of recommendations to return for each category. Default is 5.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the selected product info and recommendations.
            The dictionary has the following structure:
            {
                'selected_product': {
                    'product_id': str,
                    'image_path': str,
                    'attributes': dict,
                },
                'worn_with': List[dict],
                'complemented': List[dict],
            }

            Each recommendation in 'worn_with' and 'complemented' is a dictionary with keys:
            - 'product_id': str
            - 'image_path': str
            - 'weight': int
            - 'images': List[str]
            - 'attributes': dict
        """
        if filters is None:
            filters = {}
        recommendations = self.graph_db.get_recommendations(
            selected_product_id, filters, threshold, top_k
        )
        # Get image paths and other info for the selected product
        selected_product_image = self.product_image_map.get(selected_product_id)

        # Prepare recommendations
        def process_recommendations(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        worn_with_recs = process_recommendations(recommendations.get("worn_with", []))
        complemented_recs = process_recommendations(
            recommendations.get("complemented", [])
        )
        return {
            "selected_product": {
                "product_id": selected_product_id,
                "image_path": selected_product_image,
                "attributes": recommendations.get("selected_results", [{}])[0],
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
    ) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Process an image to find matching products from the catalog.

        The method processes the image (provided as a local path or URL), segments it into individual items,
        extracts embeddings and attributes, and performs a vector similarity search to find the closest matching
        products from the catalog for each item.

        Parameters
        ----------
        image_path_or_url : str
            Path to the image to process or URL of the image.
        visualize : bool
            Whether to visualize the segmented items and save the images.
        image_id : str, optional
            Unique identifier for the image used to name the output images. Default is an empty string.
        similarity_threshold : float, optional
            Threshold for similarity matching. Only matches with scores >= this threshold will be included.
            Default is 0.7.
        top_k : int, optional
            Number of top matches to return per item. Default is 1.

        Returns
        -------
        matched_products : List[Dict[str, Any]]
            List of matched products. Each product is a dictionary with keys:
                - 'product_id' (str): Unique identifier of the matched catalog product.
                - 'image_path' (str): Path to the product image.
                - 'attributes' (dict): Product metadata and attributes.
                - 'similarity_score' (float): Matching similarity score.
                - 'item_type' (str): Type of the item.
        filepaths : List[Optional[str]]
            List of file paths for segmented item images if visualization is enabled; otherwise, None values.

        Notes
        -----
        The function uses vector similarity search with filtering based on item type and gender.
        Items without a valid 'type' attribute will be skipped.
        """
        # Process the image
        items, filepaths = self.processor.process_image(
            image_path_or_url, visualize=visualize, image_id=image_id
        )

        matched_products = []
        for item in items:
            # Get embedding for the item
            embedding = item["embedding"]
            # Get 'type' extracted by the model
            attributes = item.get("attributes") or {}
            item_type = attributes.get("type")
            if not item_type:
                logger.warning(f"No 'type' found for item in image {image_path_or_url}")
                continue  # Skip this item if 'type' is missing
            # Set filters to retrieve items with the same 'type' and gender
            gender = attributes.get("gender")
            filters = {
                "type": item_type,
                "gender": {"$in": ["unisex", gender] if gender else ["unisex"]},
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
                        f"Similarity with product_id: {match['id']} and score: {similarity_score:.4f} for item type: {item_type}"
                    )
                    if similarity_score >= similarity_threshold:
                        catalog_product_id = match["id"]
                        logger.info(
                            f"Matching catalog item found: {catalog_product_id} for item type: {item_type}"
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
                logger.info(
                    f"No matching catalog item found for item with type '{item_type}'"
                )
        return matched_products, filepaths

    def get_outfit_from_text(
        self,
        text: str,
        text_similarity_threshold: float = 0.2,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get outfit recommendations based on a textual description.

        Parameters
        ----------
        text : str
            The textual description of the outfit.

        Returns
        -------
        List[Dict[str, Any]]
            List of matched products. Each product is a dictionary with keys:
                - 'product_id' (str): Unique identifier of the matched catalog product.
                - 'image_path' (str): Path to the product image.
                - 'attributes' (dict): Product metadata and attributes.
                - 'similarity_score' (float): Matching similarity score.
        """
        # Process the text to extract relevant keywords

        text_embedding = embedding_model.get_embedding(text=text, image=None, type="text")
        # Query vector database to find closest catalog items
        query_result = self.vector_db.query(
            text_embedding,
            top_k=top_k,
            namespace="catalog",
            include_values=True,
        )
        matched_products = []
        if query_result and "matches" in query_result and query_result["matches"]:
            for match in query_result["matches"]:
                similarity_score = match["score"]
                logger.debug(
                    f"Similarity with product_id: {match['id']} and score: {similarity_score:.4f} for text: {text}"
                )
                if similarity_score >= text_similarity_threshold:
                    catalog_product_id = match["id"]
                    logger.info(
                        f"Matching catalog item found: {catalog_product_id} for text: {text}"
                    )
                    metadata = match["metadata"]
                    product_info = {
                        "product_id": catalog_product_id,
                        "image_path": self.product_image_map.get(catalog_product_id),
                        "attributes": metadata,
                        "similarity_score": similarity_score,
                    }
                    matched_products.append(product_info)
            else:
                logger.info(f"No matching catalog item found for text: '{text}'")
        return matched_products
