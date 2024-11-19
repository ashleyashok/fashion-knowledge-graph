import os
import pandas as pd
from typing import Dict, List, Optional

from tqdm import tqdm

from src.database.graph_database import GraphDatabaseHandler
from src.database.vector_database import VectorDatabase
from src.models.model_manager import image_processor


def process_social_media_images(
    image_paths: Optional[List[str]] = None,
    image_paths_file: Optional[str] = None,
    vector_db: VectorDatabase = None,
    graph_db: Optional[GraphDatabaseHandler] = None,
    similarity_threshold: float = 0.75,
    product_type_map: Dict[str, str] = None,
    skip_attribute_extraction: bool = False,
):
    """
    Process social media images, map items to catalog products,
    and update the graph database with co-occurrence relationships.

    Parameters:
    - image_paths (Optional[List[str]]): List of image paths to process.
    - image_paths_file (Optional[str]): Path to a file containing image paths.
    - vector_db (VectorDatabase): The vector database instance.
    - graph_db (Optional[GraphDatabaseHandler]): The graph database handler.
    - similarity_threshold (float): Threshold for similarity matching.
    - product_type_map (Dict[str, str]): Mapping of product_id to product type.
    - skip_attribute_extraction (bool): Whether to skip attribute extraction.

    Note: Either image_paths or image_paths_file must be provided.
    """

    # Load image paths from list or file
    if image_paths is not None:
        # Use provided list of image paths
        image_paths_list = image_paths
    elif image_paths_file is not None:
        # Read image paths from the file
        with open(image_paths_file, "r") as f:
            image_paths_list = [line.strip() for line in f.readlines()]
    else:
        raise ValueError("Either image_paths or image_paths_file must be provided.")

    print(f"Number of outfits to be processed: {len(image_paths_list)}")

    for image_path in tqdm(image_paths_list):
        try:
            print(f"Processing social media image {image_path}")
            items, _ = image_processor.process_image(
                image_path, skip_attribute_extraction=skip_attribute_extraction
            )

            mapped_product_ids = []
            for item in items:
                # Get embedding for the item
                embedding = item["embedding"]
                # Get 'type' extracted by GPT-4O model
                attributes = item.get("attributes") or {}
                item_type = attributes.get("type")
                if not item_type:
                    print(f"No 'type' found for item in image {image_path}")
                    continue  # Skip this item if 'type' is missing
                # Set filters to retrieve items with the same 'type'
                filters = {
                    "type": item_type,
                    "gender": {"$in": ["unisex", attributes.get("gender")]},
                }
                query_result = vector_db.query(
                    embedding,
                    top_k=1,
                    namespace="catalog",
                    include_values=False,
                    filters=filters,
                )
                if (
                    query_result
                    and "matches" in query_result
                    and query_result["matches"]
                ):
                    match = query_result["matches"][0]
                    similarity_score = match["score"]
                    if similarity_score >= similarity_threshold:
                        catalog_product_id = match["id"]
                        mapped_product_ids.append(catalog_product_id)
                        print(
                            "Matching catalog item found: ",
                            catalog_product_id,
                            ": ",
                            item_type,
                        )
                    else:
                        print(
                            f"No matching catalog item found for item with type '{item_type}' "
                            f"and sufficient similarity (score: {similarity_score})"
                        )
                else:
                    print(
                        f"No matching catalog item found for item with type '{item_type}'"
                    )
            # Update graph database with co-occurrence relationships
            if len(mapped_product_ids) > 1:
                # Extract the image file name from the image path
                image_file_name = os.path.basename(image_path)
                # Update relationships in the graph database
                for i in range(len(mapped_product_ids)):
                    for j in range(i + 1, len(mapped_product_ids)):
                        product_id1 = mapped_product_ids[i]
                        product_id2 = mapped_product_ids[j]
                        # Determine relationship type based on product types
                        type1 = product_type_map.get(product_id1)
                        type2 = product_type_map.get(product_id2)
                        if type1 and type2:
                            if type1 == type2:
                                relationship_type = "COMPLEMENTED_BY"
                            else:
                                relationship_type = "WORN_WITH"
                        else:
                            # Default to WORN_WITH if types are missing
                            relationship_type = "WORN_WITH"
                        # Include the image file name in the edge properties
                        properties = {"image": image_file_name}
                        graph_db.create_or_update_relationship(
                            product_id1,
                            product_id2,
                            relationship_type,
                            properties=properties,
                        )
                        graph_db.create_or_update_relationship(
                            product_id2,
                            product_id1,
                            relationship_type,
                            properties=properties,
                        )
        except Exception as e:
            print(f"Error processing social media image {image_path}: {e}")
            continue


if __name__ == "__main__":
    vector_db = VectorDatabase(
        index_name="catalog-clothes",
    )
    graph_db = GraphDatabaseHandler(
        uri=os.getenv("NEO4J_URI"),
        user=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD"),
    )

    catalog_df = pd.read_csv("output/data/catalog5_celebrity.csv")
    catalog_df["product_id"] = catalog_df["product_id"].astype(str)
    product_type_map = catalog_df.set_index("product_id")["category"].to_dict()

    # Process social media images
    process_social_media_images(
        # "output/data/social_media_images.txt",
        image_paths=["dataset/celebrity_outfits/celebrity_1.jpg"],
        vector_db=vector_db,
        graph_db=graph_db,
        product_type_map=product_type_map,
        skip_attribute_extraction=False,
    )
    # Close graph database connection
    graph_db.close()
