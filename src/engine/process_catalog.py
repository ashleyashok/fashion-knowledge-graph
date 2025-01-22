# src/engine/process_catalog.py

import os
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.database.graph_database import GraphDatabaseHandler
from src.database.vector_database import VectorDatabase
from src.models.model_manager import image_processor


def process_catalog(
    catalog_df: pd.DataFrame,
    vector_db_image: VectorDatabase,
    vector_db_style: VectorDatabase,
    graph_db: GraphDatabaseHandler,
):
    """
    Process the retailer catalog to generate embeddings and attributes,
    store them in the vector databases, and create nodes in the graph database.
    """
    for index, row in tqdm(catalog_df.iterrows(), total=catalog_df.shape[0]):
        logger.info(f"Processing catalog product {index + 1}/{catalog_df.shape[0]}")
        try:
            product_id = str(row["product_id"])
            image_path = row["image_path"]
            # Process the image
            logger.info(f"Processing catalog product {product_id}")
            items, _ = image_processor.process_image(
                image_path,
                image_id=product_id,
                single_product_mode=True,
                skip_attribute_extraction=False,
                # skip_attribute_extraction=True,
                skip_style_extraction=False,
            )
            # Since single product mode, assume only one item per image
            if len(items) == 0:
                logger.warning(f"No items found in image {image_path}")
                continue
            item = items[0]
            # Include product_id in metadata
            attributes = item.get("attributes") or {}
            item_metadata = attributes.copy()
            item_metadata["segmented_label"] = item["label"]
            item_metadata["product_id"] = product_id
            item_metadata["style_description"] = item.get("style_description", "")
            # Upsert image embedding to vector database with namespace 'catalog'
            vector_db_image.upsert_embeddings(
                [
                    {
                        "id": product_id,
                        "embedding": item["embedding"],
                        "metadata": item_metadata,
                    }
                ],
                namespace="catalog",
            )
            # Upsert style embedding to vector database with namespace 'catalog_style'
            if item.get("style_embedding") is not None:
                style_metadata = {
                    "style_description": item.get("style_description", ""),
                    "product_id": product_id,
                }
                vector_db_style.upsert_embeddings(
                    [
                        {
                            "id": product_id,
                            "embedding": item["style_embedding"],
                            "metadata": style_metadata,
                        }
                    ],
                    namespace="catalog_style",
                )
            # Create product node in graph database
            graph_db.create_product_node(
                product_id=product_id, attributes=item_metadata
            )
        except Exception as e:
            logger.error(f"Error processing catalog product {product_id}: {e}")
            continue



def main():
    # Initialize vector database and graph database
    vector_db_image = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_IMAGE"),
        index_name="catalog-clothes",
    )
    vector_db_style = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_STYLE"),
        index_name="catalog-style-description",
    )

    graph_db = GraphDatabaseHandler(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    # Load catalog data
    catalog_df = pd.read_csv("output/data/catalog8_gq.csv")
    catalog_df["product_id"] = catalog_df["product_id"].astype(str)
    process_catalog(catalog_df, vector_db_image, vector_db_style, graph_db)

if __name__ == "__main__":
    main()
