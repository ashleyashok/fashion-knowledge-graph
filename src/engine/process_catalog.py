# src/preprocessing/process_catalog.py

from typing import List, Optional, Dict
import pandas as pd
from src.database.vector_database import VectorDatabase
from src.database.graph_database import GraphDatabaseHandler
import os
from tqdm import tqdm
from src.models.model_manager import image_processor


def process_catalog(
    catalog_df: pd.DataFrame, vector_db: VectorDatabase, graph_db: GraphDatabaseHandler
):
    """
    Process the retailer catalog to generate embeddings and attributes,
    store them in the vector database, and create nodes in the graph database.
    """
    for index, row in tqdm(catalog_df.iterrows(), total=catalog_df.shape[0]):
        try:
            product_id = str(row["product_id"])
            image_path = row["image_path"]
            # Process the image
            print(f"Processing catalog product {product_id}")
            items, _ = image_processor.process_image(
                image_path, image_id=product_id, single_product_mode=True
            )
            # Since single product mode, assume only one item per image
            if len(items) == 0:
                print(f"No items found in image {image_path}")
                continue
            item = items[0]
            # Include product_id in metadata
            item_metadata = item["attributes"].attributes
            item_metadata["segmented_label"] = item["label"]
            item_metadata["product_id"] = product_id
            # Upsert to vector database with namespace 'catalog'
            vector_db.upsert_embeddings(
                [
                    {
                        "id": product_id,
                        "embedding": item["embedding"],
                        "metadata": item_metadata,
                    }
                ],
                namespace="catalog",
            )
            # Create product node in graph database
            graph_db.create_product_node(
                product_id=product_id, attributes=item_metadata
            )
        except Exception as e:
            print(f"Error processing catalog product {product_id}: {e}")
            continue



def main():
    # Initialize vector database and graph database
    vector_db = VectorDatabase(
        index_name="catalog-clothes",
    )
    graph_db = GraphDatabaseHandler(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )
    # Load catalog data
    catalog_df = pd.read_csv("output/data/catalog6_macy.csv")
    catalog_df["product_id"] = catalog_df["product_id"].astype(str)
    process_catalog(catalog_df, vector_db, graph_db)

if __name__ == "__main__":
    main()
