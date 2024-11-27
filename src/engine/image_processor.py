# src/engine/image_processor.py

import os
from typing import List, Dict, Optional, Tuple, Any

from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from src.models.segmentation_model import SegmentationModel

# from src.models.model_manager import embedding_model
from src.models.embedding_model import BaseEmbeddingModel, ClipEmbeddingModel
from src.models.attribute_extraction_model import AttributeExtractionModel
from src.utils.models import ProductAttributes, EmbeddingData


class ImageProcessor:
    """
    A class to process images, segment clothing items, extract attributes, and generate embeddings.
    """

    def __init__(
        self,
        segmentation_model: SegmentationModel,
        embedding_model: BaseEmbeddingModel,
        text_embedding_model: BaseEmbeddingModel,
        attribute_model: AttributeExtractionModel,
        visualize_dir: Optional[str] = None,
    ):
        """
        Initialize the ImageProcessor.

        Parameters
        ----------
        segmentation_model : SegmentationModel
            An instance of SegmentationModel.
        embedding_model : BaseEmbeddingModel
            An instance of EmbeddingModel for image embeddings.
        text_embedding_model : BaseEmbeddingModel
            An instance of EmbeddingModel for text embeddings.
        attribute_model : AttributeExtractionModel
            An instance of AttributeExtractionModel.
        visualize_dir : str, optional
            Directory to save visualizations. If None, visualizations are not saved.
        """
        self.segmentation_model = segmentation_model
        self.embedding_model = embedding_model
        self.text_embedding_model = text_embedding_model
        self.attribute_model = attribute_model

        # Retrieve id2label mapping from segmentation model
        self.id2label = self.segmentation_model.get_id2label()

        # Visualization directory
        self.visualize_dir = visualize_dir
        if self.visualize_dir and not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
            logger.info(f"Created visualization directory at {self.visualize_dir}")

    def load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from a local path or URL.

        Parameters
        ----------
        image_path_or_url : str
            The path or URL of the image to load.

        Returns
        -------
        PIL.Image.Image
            The loaded image.
        """
        logger.info(f"Loading image from {image_path_or_url}")
        try:
            if image_path_or_url.startswith("http"):
                response = requests.get(image_path_or_url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path_or_url).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            raise

    def segment_image(
        self,
        image: Image.Image,
        visualize: bool = False,
        image_id: str = "",
        single_product_mode: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Segments the given image to extract individual clothing items.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Optional[str]]]
            A tuple containing:
                - A list of dictionaries, each containing 'label', 'image', and 'area' for each extracted item.
                - A list of file paths where the visualized segments are saved (if visualize is True).
        """
        logger.info("Processing image for segmentation...")
        try:
            seg = self.segmentation_model.segment(image)
            # Extract individual clothing items
            items = []
            filepaths = []
            logger.info("Extracting individual clothing items...")
            for class_id, label in self.id2label.items():
                if class_id == 0:
                    continue  # Skip background class
                mask = seg.cpu().numpy() == class_id
                area = mask.sum()
                if area < 1028:
                    continue  # Skip if the item is not present or too small

                # Create a mask image
                mask_image = Image.fromarray((mask * 255).astype("uint8"))
                # Apply mask to the original image
                item_image = Image.composite(
                    image, Image.new("RGB", image.size), mask_image
                )
                items.append({"label": label, "image": item_image, "area": area})

                logger.debug(f"Extracted item: {label} with area {area}")

                # Visualize the segment if requested
                if visualize:
                    filepath = self.visualize_segment(item_image, label, image_id)
                    filepaths.append(filepath)
                else:
                    filepaths.append(None)

            if single_product_mode and items:
                # Sort items by area in descending order and keep the largest one
                items = sorted(items, key=lambda x: x["area"], reverse=True)
                items = [items[0]]
                logger.info("Single product mode: selected the largest segment.")

            logger.info(f"Total items extracted: {len(items)}")
            return items, filepaths
        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise

    def visualize_segment(
        self, item_image: Image.Image, label: str, image_id: str
    ) -> Optional[str]:
        """
        Visualize and save the segmented item.

        Parameters
        ----------
        item_image : PIL.Image.Image
            The segmented item image.
        label : str
            The label of the item.
        image_id : str
            An identifier for the image.

        Returns
        -------
        Optional[str]
            The file path where the image is saved, or None if not saved.
        """
        try:
            if self.visualize_dir:
                filename = f"{image_id}_{label.replace(',', '').replace(' ', '_')}.png"
                filepath = os.path.join(self.visualize_dir, filename)
                item_image.save(filepath)
                logger.info(f"Saved visualization: {filepath}")
                return filepath
        except Exception as e:
            logger.error(f"Error visualizing segment: {e}")
        return None

    def process_image(
        self,
        image_path_or_url: str,
        visualize: bool = False,
        image_id: str = "",
        single_product_mode: bool = False,
        skip_attribute_extraction: bool = False,
        skip_style_extraction: bool = False,
    ) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """
        Process an image to extract items, embeddings, attributes, and style descriptions.

        Returns
        -------
        Tuple[List[Dict[str, Any]], List[Optional[str]]]
            A tuple containing:
                - List of dictionaries with processed items, each containing:
                    - 'label': Item classification label
                    - 'attributes': Extracted attributes (None if skipped)
                    - 'style_description': Style description (empty if skipped)
                    - 'embedding': Vector embedding of the item (image embedding)
                    - 'style_embedding': Embedding of the style description (text embedding)
                - List of file paths for any saved visualizations
        """
        logger.info(f"Processing image: {image_path_or_url}")
        try:
            image = self.load_image(image_path_or_url)
            items, filepaths = self.segment_image(
                image,
                visualize=visualize,
                image_id=image_id,
                single_product_mode=single_product_mode,
            )
            results = []

            for item in items:
                attributes = None
                style_description = ""
                style_embedding = None
                if not skip_attribute_extraction:
                    attributes = self.attribute_model.extract_attributes(
                        item["image"], item["label"]
                    )
                if not skip_style_extraction:
                    style_description = self.attribute_model.extract_style_description_from_image(
                        item["image"], item["label"]
                    )
                    # Generate embedding for the style description
                    style_embedding = self.text_embedding_model.get_embedding(
                        text=style_description, type="text"
                    )
                embedding = self.embedding_model.get_embedding(
                    image=item["image"], text=item["label"], type="image"
                )
                result = {
                    "label": item["label"],
                    "attributes": attributes,
                    "style_description": style_description,
                    "embedding": embedding,
                    "style_embedding": style_embedding,
                }
                results.append(result)

                logger.info(f"Processed item: {item['label']}")
                logger.debug(f"Attributes: {attributes}")
                logger.debug(f"Style Description: {style_description}")
                if style_embedding is not None:
                    logger.debug(f"Style Embedding size: {len(style_embedding)}")
                logger.debug(f"Embedding size: {len(embedding)}")

            logger.info("Finished processing image.")
            return results, filepaths
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise


if __name__ == "__main__":
    from pprint import pprint
    from src.models.embedding_model import (
        VertexAIEmbeddingModel,
        SentenceTransformerEmbeddingModel,
    )
    from src.utils.tools import cosine_similarity, euclidean_distance
    import vertexai
    from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage
    import numpy as np

    id2label = {
        0: "Background",
        3: "Sunglasses",
        4: "Upper-clothes",
        5: "Skirt",
        6: "Pants",
        7: "Dress",
        8: "Belt",
        16: "Bag",
        17: "Scarf",
    }

    # Initialize models (device is optional)
    segmentation_model = SegmentationModel(
        model_name="sayeed99/segformer_b3_clothes",
        id2label=id2label,
    )

    # embedding_model = VertexAIEmbeddingModel()
    embedding_model = ClipEmbeddingModel(
        model_name="Marqo/marqo-fashionCLIP",
    )

    attribute_model = AttributeExtractionModel()

    text_embedding_model = SentenceTransformerEmbeddingModel(
        model_name="all-MiniLM-L6-v2",
    )

    # Initialize ImageProcessor
    processor = ImageProcessor(
        segmentation_model=segmentation_model,
        embedding_model=embedding_model,
        attribute_model=attribute_model,
        text_embedding_model=text_embedding_model,
        visualize_dir="temp_images/segmented_images",
    )

    # Process an image
    results, filepaths = processor.process_image(
        image_path_or_url="dataset/macy_clothes/macy_16b.jpg",
        visualize=True,
        image_id="macy_barbie",
        single_product_mode=True,
        skip_attribute_extraction=False,
        skip_style_extraction=False,
    )

    print(results)

    # for result in results:
    #     if result["label"] == "Dress":
    #         v_top = result["embedding"]
    #         break

    # from transformers import AutoModel, AutoProcessor

    # model = AutoModel.from_pretrained("Marqo/marqo-fashionCLIP", trust_remote_code=True)
    # processor = AutoProcessor.from_pretrained(
    #     "Marqo/marqo-fashionCLIP", trust_remote_code=True
    # )

    # import torch
    # from PIL import Image

    # image = [Image.open("dataset/macy_clothes/macy_11.jpg")]
    # text = ["photo of barbie clothing"]
    # processed = processor(
    #     text=text, images=image, padding="max_length", return_tensors="pt"
    # )

    # with torch.no_grad():
    #     image_features = model.get_image_features(
    #         processed["pixel_values"], normalize=True
    #     )
    #     image_features = image_features.cpu().numpy()[0].tolist()
    #     text_features = model.get_text_features(processed["input_ids"], normalize=True)
    #     text_features = text_features.cpu().numpy()[0].tolist()

    # PROJECT_ID = "gemini-copilot-testing"
    # vertexai.init(project=PROJECT_ID, location="us-central1")

    # model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    # image = VertexImage.load_from_file("dataset/celebrity_clothes/macy_1.jpeg")

    # embeddings = model.get_embeddings(
    #     image=image,
    #     contextual_text="vintage themed clothing",
    #     dimension=1408,
    # )

    # v_text = embeddings.text_embedding

    # # single product embedding
    # v_top = results[0]["embedding"]

    # # Process an image
    # results, filepaths = processor.process_image(
    #     image_path_or_url="dataset/celebrity_outfits/celebrity_4.jpg",
    #     visualize=True,
    #     image_id="celebrity_4_top1",
    #     single_product_mode=False,
    #     skip_attribute_extraction=False,
    # )

    # v_outfit_top = results[0]["embedding"]

    # Calculate cosine similarity
    # print(text_features)
    # similarity = cosine_similarity(np.array(v_top), np.array(text_features).flatten())
    # print(f"Cosine similarity between top and outfit: {similarity}")
