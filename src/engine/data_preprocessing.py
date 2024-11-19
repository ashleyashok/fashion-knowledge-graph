# src/preprocessing/data_preprocessing.py

import os
from typing import List, Set
import pandas as pd

def preprocess_retail_catalog(
    retail_data_path: str,
    image_dir: str,
    combined_catalog_path: str,
    catalog_output_path: str,
    include_categories: List[str] = ['tops', 'all-body'],
    n_samples: int = 500
) -> pd.DataFrame:
    """
    Process the Polyvore dataset to create the retail catalog DataFrame.

    Parameters:
    - retail_data_path (str): Path to the retail data JSON file.
    - image_dir (str): Directory where images are stored.
    - combined_catalog_path (str): Path to the combined catalog CSV file.
    - catalog_output_path (str): Path where the new catalog CSV will be saved.
    - include_categories (List[str], optional): List of semantic categories to include. Defaults to ['tops', 'all-body'].
    - n_samples (int, optional): Number of samples to select from each category. Defaults to 500.

    Returns:
    - pd.DataFrame: The processed catalog DataFrame.
    """
    # Load retail data JSON as pandas DataFrame
    retail_data_metadata = pd.read_json(retail_data_path)
    retail_data_metadata = retail_data_metadata.transpose()
    retail_data_metadata.reset_index(inplace=True)  # 'index' column contains item IDs

    # Include only specified categories
    retail_data_metadata_filt = retail_data_metadata[
        retail_data_metadata['semantic_category'].isin(include_categories)
    ]

    # Exclude items whose product_id already exists in the combined catalog
    if os.path.exists(combined_catalog_path):
        catalog_combined = pd.read_csv(combined_catalog_path)
        existing_product_ids: Set[str] = set(catalog_combined['product_id'].astype(str))
    else:
        existing_product_ids = set()

    # Remove items with product_id in existing_product_ids
    retail_data_metadata_filt = retail_data_metadata_filt[
        ~retail_data_metadata_filt['index'].astype(str).isin(existing_product_ids)
    ]

    # Get N number of samples from each semantic category
    retail_data_metadata_sampled = retail_data_metadata_filt.groupby('semantic_category').apply(
        lambda x: x.sample(n=min(len(x), n_samples), random_state=42)
    ).reset_index(drop=True)

    # Construct the image paths
    retail_data_metadata_sampled['image_path'] = retail_data_metadata_sampled['index'].apply(
        lambda x: os.path.join(image_dir, f"{x}.jpg")
    )

    # Check if images exist
    retail_data_metadata_sampled = retail_data_metadata_sampled[
        retail_data_metadata_sampled['image_path'].apply(os.path.exists)
    ]

    # Create a DataFrame with required columns for the pipeline
    catalog_df = retail_data_metadata_sampled[['index', 'image_path', 'semantic_category']]
    catalog_df.rename(columns={'index': 'product_id', 'semantic_category': 'category'}, inplace=True)

    # Save the catalog_df to the specified output path
    catalog_df.to_csv(catalog_output_path, index=False)
    print(f"Saved new catalog data to {catalog_output_path}")

    # Append catalog_df to the combined catalog
    if os.path.exists(combined_catalog_path):
        catalog_combined = pd.read_csv(combined_catalog_path)
        catalog_combined = pd.concat([catalog_combined, catalog_df], ignore_index=True)
        # Remove duplicates based on 'product_id'
        catalog_combined.drop_duplicates(subset='product_id', inplace=True)
        print(f"Appended new data to existing {combined_catalog_path}")
    else:
        catalog_combined = catalog_df
        print(f"Created new combined catalog at {combined_catalog_path}")

    # Save updated combined catalog
    catalog_combined.to_csv(combined_catalog_path, index=False)
    print(f"Updated combined catalog saved to {combined_catalog_path}")

    return catalog_df

def preprocess_social_media_images():
    """
    Process the DeepFashion dataset to get a list of social media image paths.
    """
    social_media_dir = "dataset/DeepFashion"
    all_files = os.listdir(social_media_dir)
    # Filter files ending with '_full.jpg'
    full_images = [f for f in all_files if f.endswith('_full.jpg')]
    # Select 100 images
    n_images = 1000
    selected_images = full_images[:n_images]
    # Construct full image paths
    image_paths = [os.path.join(social_media_dir, f) for f in selected_images]
    return image_paths

if __name__ == "__main__":
    # Retail catalog settings
    retail_data_path = "dataset/polyvore_outfits/polyvore_item_metadata.json"
    image_dir = "dataset/polyvore_outfits/images"
    combined_catalog_path = "output/data/catalog_combined.csv"
    catalog_output_path = "output/data/catalog3.csv"
    include_categories = ['tops', 'all-body']
    n_samples = 100

    # Social media images settings
    social_media_dir = "dataset/DeepFashion"
    image_list_output_path = "output/data/social_media_images.txt"
    n_social_media_images = 1000

    # Process retail catalog data
    catalog_df = preprocess_retail_catalog(
        retail_data_path=retail_data_path,
        image_dir=image_dir,
        combined_catalog_path=combined_catalog_path,
        catalog_output_path=catalog_output_path,
        include_categories=include_categories,
        n_samples=n_samples
    )
    print("Retail Catalog Data:")
    print(catalog_df.head())


    # Save the catalog data to a CSV file
    # catalog_df.to_csv('output/data/catalog2.csv', index=False)

    # Process social media images 
    # social_media_image_paths = preprocess_social_media_images()
    # print("Social Media Image Paths:")
    # print(social_media_image_paths[:5])

    # # Save the social media image paths to a file
    # with open('output/data/social_media_images.txt', 'w') as f:
    #     for path in social_media_image_paths:
    #         f.write(f"{path}\n")
