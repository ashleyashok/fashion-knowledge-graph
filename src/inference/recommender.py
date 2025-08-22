"""
Recommendation engine for the Complete the Look fashion recommendation system.

This module provides the Recommender class that combines graph-based
and vector-based approaches to generate fashion recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger

from src.database.graph_database import GraphDatabaseHandler
from src.database.vector_database import VectorDatabase
from src.models.model_manager import model_manager


class Recommender:
    """
    Fashion recommendation engine.

    This class combines knowledge graph traversal and vector similarity
    search to provide comprehensive fashion recommendations. It can
    generate recommendations based on product relationships, image
    similarity, and textual descriptions.
    """

    def __init__(
        self,
        graph_db: GraphDatabaseHandler,
        catalog_csv_path: str,
        vector_db_image: VectorDatabase,
        vector_db_style: VectorDatabase,
    ):
        """
        Initialize the recommendation engine.

        Parameters
        ----------
        graph_db : GraphDatabaseHandler
            Handler for Neo4j graph database operations.
        catalog_csv_path : str
            Path to the catalog CSV file containing product data.
        vector_db_image : VectorDatabase
            Handler for image embeddings in Pinecone.
        vector_db_style : VectorDatabase
            Handler for style embeddings in Pinecone.
        """
        self.graph_db = graph_db
        self.vector_db_image = vector_db_image
        self.vector_db_style = vector_db_style

        # Load and prepare catalog data
        self._load_catalog_data(catalog_csv_path)

        # Get models from model manager
        self.processor = model_manager.image_processor
        self.image_embedding_model = model_manager.embedding_model
        self.text_embedding_model = model_manager.text_embedding_model

        logger.info("Recommender initialized successfully")

    def _load_catalog_data(self, catalog_csv_path: str) -> None:
        """
        Load and prepare catalog data for quick lookups.

        Parameters
        ----------
        catalog_csv_path : str
            Path to the catalog CSV file.
        """
        try:
            logger.info(f"Loading catalog data from {catalog_csv_path}")

            self.catalog_df = pd.read_csv(catalog_csv_path)
            self.catalog_df["product_id"] = self.catalog_df["product_id"].astype(str)

            # Create mappings for quick lookup
            self.product_image_map = dict(
                zip(self.catalog_df["product_id"], self.catalog_df["image_path"])
            )
            self.product_attributes_map = self.catalog_df.set_index(
                "product_id"
            ).to_dict("index")

            logger.info(f"Loaded {len(self.catalog_df)} products from catalog")

        except Exception as e:
            logger.error(f"Failed to load catalog data: {e}")
            raise

    def get_recommendations(
        self,
        selected_product_id: str,
        filters: Optional[Dict[str, Any]] = None,
        threshold: int = 1,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Get recommended products for a selected product.

        This method uses graph traversal to find products that are
        frequently worn together or complement the selected product.

        Parameters
        ----------
        selected_product_id : str
            The product ID of the selected product.
        filters : Dict[str, Any], optional
            A dictionary of filters to apply when retrieving recommendations.
            For example, {'type': 'shirt'}. Default is None.
        threshold : int, default=1
            The minimum weight threshold for relationships to consider.
        top_k : int, default=5
            The maximum number of recommendations to return for each category.

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

        try:
            # Get recommendations from graph database
            recommendations = self.graph_db.get_recommendations(
                selected_product_id, filters, threshold, top_k
            )

            # Get image path for the selected product
            selected_product_image = self.product_image_map.get(selected_product_id)

            # Process recommendations
            def process_recommendations(
                recs: List[Dict[str, Any]],
            ) -> List[Dict[str, Any]]:
                """Process and enrich recommendation data."""
                processed = []
                for rec in recs:
                    product_id = rec["product_id"]
                    weight = rec["weight"]
                    source = rec.get("source", "")
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
                                "source": source,
                                "images": images,
                                "attributes": metadata,
                            }
                        )
                return processed

            # Process both types of recommendations
            worn_with_recs = process_recommendations(
                recommendations.get("worn_with", [])
            )
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

        except Exception as e:
            logger.error(
                f"Error getting recommendations for product {selected_product_id}: {e}"
            )
            raise

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

        This method segments the image into individual clothing items,
        extracts embeddings and attributes, and performs vector similarity
        search to find the closest matching products from the catalog.

        Parameters
        ----------
        image_path_or_url : str
            Path to the image to process or URL of the image.
        visualize : bool
            Whether to visualize the segmented items and save the images.
        image_id : str, default=""
            Unique identifier for the image used to name the output images.
        similarity_threshold : float, default=0.7
            Threshold for similarity matching. Only matches with scores >= this
            threshold will be included.
        top_k : int, default=1
            Number of top matches to return per item.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Optional[str]]]
            A tuple containing:
            - List of matched products. Each product is a dictionary with keys:
                - 'product_id' (str): Unique identifier of the matched catalog product.
                - 'image_path' (str): Path to the product image.
                - 'attributes' (dict): Product metadata and attributes.
                - 'similarity_score' (float): Matching similarity score.
                - 'item_type' (str): Type of the item.
            - List of file paths for segmented item images if visualization is enabled;
              otherwise, None values.
        """
        try:
            # Process the image to extract items
            items, filepaths = self.processor.process_image(
                image_path_or_url, visualize=visualize, image_id=image_id
            )

            matched_products = []

            for item in items:
                # Get embedding for the item
                embedding = item["embedding"]

                # Get attributes extracted by the model
                attributes = item.get("attributes") or {}
                item_type = attributes.get("type")
                color = attributes.get("color") or {}

                if not item_type:
                    logger.warning(
                        f"No 'type' found for item in image {image_path_or_url}"
                    )
                    continue  # Skip this item if 'type' is missing

                # Set filters to retrieve items with the same 'type' and gender
                gender = attributes.get("gender")
                filters = {
                    "type": item_type,
                    "gender": {"$in": ["unisex", gender] if gender else ["unisex"]},
                    "color": color,
                }

                # Query vector database for similar products
                query_result = self.vector_db_image.query(
                    embedding,
                    top_k=top_k,
                    namespace="catalog",
                    include_values=False,
                    filters=filters,
                )

                if (
                    query_result
                    and "matches" in query_result
                    and query_result["matches"]
                ):
                    for match in query_result["matches"]:
                        similarity_score = match["score"]
                        logger.debug(
                            f"Similarity with product_id: {match['id']} and score: "
                            f"{similarity_score:.4f} for item type: {item_type}"
                        )

                        if similarity_score >= similarity_threshold:
                            catalog_product_id = match["id"]
                            logger.info(
                                f"Matching catalog item found: {catalog_product_id} "
                                f"for item type: {item_type}"
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

        except Exception as e:
            logger.error(f"Error processing image {image_path_or_url}: {e}")
            raise

    def get_outfit_from_text(
        self,
        text: str,
        clip_text_similarity_threshold: float = 0.2,
        style_text_similarity_threshold: float = 0.5,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Get outfit recommendations based on a textual description.

        This method performs two types of searches:
        1. CLIP embedding search for text-to-image similarity
        2. Style embedding search for semantic similarity

        Results are combined using Rank Reciprocal Fusion (RRF).

        Parameters
        ----------
        text : str
            The textual description of the outfit.
        clip_text_similarity_threshold : float, default=0.2
            Threshold for CLIP text-to-image similarity matching.
        style_text_similarity_threshold : float, default=0.5
            Threshold for style text similarity matching.
        top_k : int, default=5
            Number of top matches to return from each index.
        rrf_k : int, default=60
            The constant k in the Rank Reciprocal Fusion (RRF) formula.

        Returns
        -------
        List[Dict[str, Any]]
            List of matched products combined from both searches and sorted by fused scores.
        """
        try:
            # First Search: CLIP Embedding Search in Image Embeddings Index
            matched_products_clip = self._search_clip_embeddings(
                text, clip_text_similarity_threshold, top_k
            )

            # Second Search: Style Embedding Search in Style Embeddings Index
            matched_products_style = self._search_style_embeddings(
                text, style_text_similarity_threshold, top_k
            )

            # Combine Results Using Rank Reciprocal Fusion (RRF)
            combined_results = self._rank_reciprocal_fusion(
                [matched_products_clip, matched_products_style], k=rrf_k
            )

            return combined_results[:top_k]

        except Exception as e:
            logger.error(f"Error getting outfit from text '{text}': {e}")
            raise

    def _search_clip_embeddings(
        self, text: str, similarity_threshold: float, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for products using CLIP embeddings.

        Parameters
        ----------
        text : str
            Text description to search for.
        similarity_threshold : float
            Minimum similarity threshold.
        top_k : int
            Number of top results to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of matched products.
        """
        # Get the CLIP embedding of the input text
        text_embedding_clip = self.image_embedding_model.get_embedding(
            text=text, type="text"
        )

        # Query the image embeddings index
        query_result_clip = self.vector_db_image.query(
            embedding=text_embedding_clip,
            top_k=top_k,
            namespace="catalog",
            include_values=False,
        )

        # Process results
        matched_products = []
        if (
            query_result_clip
            and "matches" in query_result_clip
            and query_result_clip["matches"]
        ):
            for match in query_result_clip["matches"]:
                similarity_score = match["score"]
                if similarity_score >= similarity_threshold:
                    catalog_product_id = match["id"]
                    metadata = match["metadata"]
                    product_info = {
                        "product_id": catalog_product_id,
                        "image_path": self.product_image_map.get(catalog_product_id),
                        "attributes": metadata,
                        "clip_similarity_score": similarity_score,
                    }
                    matched_products.append(product_info)
        else:
            logger.info(
                f"No matching catalog item found for text: '{text}' in image embeddings"
            )

        logger.info(f"Found {len(matched_products)} products from CLIP search")
        return matched_products

    def _search_style_embeddings(
        self, text: str, similarity_threshold: float, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search for products using style embeddings.

        Parameters
        ----------
        text : str
            Text description to search for.
        similarity_threshold : float
            Minimum similarity threshold.
        top_k : int
            Number of top results to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of matched products.
        """
        # Extract style description from text
        style_description = (
            model_manager.attribute_model.extract_style_description_from_text(text)
        )

        # Get the embedding of the style description
        style_embedding = self.text_embedding_model.get_embedding(
            text=style_description, type="text"
        )

        # Query the style embeddings index
        query_result_style = self.vector_db_style.query(
            embedding=style_embedding,
            top_k=top_k,
            namespace="catalog_style",
            include_values=False,
        )

        # Process results
        matched_products = []
        if (
            query_result_style
            and "matches" in query_result_style
            and query_result_style["matches"]
        ):
            for match in query_result_style["matches"]:
                similarity_score = match["score"]
                if similarity_score >= similarity_threshold:
                    catalog_product_id = match["id"]
                    metadata = match["metadata"]
                    product_info = {
                        "product_id": catalog_product_id,
                        "image_path": self.product_image_map.get(catalog_product_id),
                        "attributes": metadata,
                        "style_description": metadata.get("style_description", ""),
                        "style_similarity_score": similarity_score,
                    }
                    matched_products.append(product_info)
        else:
            logger.info(
                f"No matching catalog item found for text: '{text}' in style embeddings"
            )

        logger.info(
            f"Found {len(matched_products)} products from style description search"
        )
        return matched_products

    def _rank_reciprocal_fusion(
        self, result_lists: List[List[Dict[str, Any]]], k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Combine multiple result lists using Rank Reciprocal Fusion (RRF).

        Parameters
        ----------
        result_lists : List[List[Dict[str, Any]]]
            A list of result lists to combine. Each result list is a list of product dicts.
        k : int, default=60
            The constant k in the RRF formula.

        Returns
        -------
        List[Dict[str, Any]]
            Combined list of products sorted by their fused scores.
        """
        scores = {}

        for result_list in result_lists:
            for rank, item in enumerate(result_list, start=1):
                product_id = item["product_id"]
                score = 1 / (k + rank)

                if product_id in scores:
                    scores[product_id]["score"] += score
                else:
                    scores[product_id] = {
                        "product": item,
                        "score": score,
                    }

        # Sort the items by their fused scores
        combined_results = sorted(
            scores.values(), key=lambda x: x["score"], reverse=True
        )

        # Extract the product info
        combined_products = [entry["product"] for entry in combined_results]
        return combined_products
