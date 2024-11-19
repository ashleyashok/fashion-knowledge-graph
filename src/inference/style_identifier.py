# Temp file for internal testing

from pprint import pprint
import os

from src.database.graph_database import GraphDatabaseHandler
from src.engine.image_processor import ImageProcessor
from src.database.vector_database import VectorDatabase

similarity_threshold = 0.75

vector_db = VectorDatabase(
    index_name="catalog-clothes",
)
# graph_db = GraphDatabaseHandler(
#     uri=os.getenv("NEO4J_URI"),
#     user=os.getenv("NEO4J_USERNAME", "neo4j"),
#     password=os.getenv("NEO4J_PASSWORD"),
# )


processor = ImageProcessor(visualize_dir="temp_images")
items, segmented_filepaths = processor.process_image(
    image_path_or_url="dataset/celebrity_outfits/celebrity_4.jpg",
    visualize=True,
    image_id="celebrity4",
    single_product_mode=False,
    skip_attribute_extraction=False,
)
pprint(segmented_filepaths)

mapped_product_ids = []
unmapped_product_ids = []
for item in items:
    # Get embedding for the item
    embedding = item["embedding"].embedding
    # Query vector database to find closest catalog item
    # Apply filters to retrieve items with the same segmented label
    item_type = item["attributes"].attributes.get("type")
    if not item_type:
        print(f"No 'type' found for item in image {item}")
        continue  # Skip this item if 'type' is missing
    # Set filters to retrieve items with the same 'type'

    filters = {
        "type": item_type,
        "gender": {"$in": ["unisex", item["attributes"].attributes.get("gender")]},
    }
    print(filters)
    query_result = vector_db.query(
        embedding,
        top_k=1,
        namespace="catalog",
        include_values=False,
        filters=filters,
    )
    if query_result and "matches" in query_result and query_result["matches"]:
        match = query_result["matches"][0]
        similarity_score = match["score"]
        unmapped_product_ids.append(match["id"])
        if similarity_score >= similarity_threshold:
            catalog_product_id = match["id"]
            mapped_product_ids.append(catalog_product_id)
            print(
                "Matching catalog item found: ",
                catalog_product_id,
                ": ",
                item["label"],
            )
        else:
            print(
                f"No matching catalog item found for item {item['label']} with sufficient similarity (score: {similarity_score})"
            )
    else:
        print(f"No matching catalog item found for item {item['label']}")

print("unmapped_product_ids:", unmapped_product_ids)
