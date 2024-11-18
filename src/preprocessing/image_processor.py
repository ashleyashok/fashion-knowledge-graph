# src/preprocessing/image_processor.py

import os
import sys
import base64
from openai import AzureOpenAI
import requests
from io import BytesIO
from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt

from transformers import (
    SegformerImageProcessor,
    AutoModelForSemanticSegmentation,
    AutoModel,
    AutoProcessor,
)

from src.utils.models import ProductAttributes, EmbeddingData
from src.utils.prompts import ATTRIBUTES_TO_EXTRACT_PROMPT


class ImageProcessor:
    """
    A class to process images, segment clothing items, extract attributes, and generate embeddings.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        visualize_dir: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load the segmentation model
        logger.info("Loading segmentation model...")
        self.segmentation_processor = SegformerImageProcessor.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        )
        self.segmentation_model = AutoModelForSemanticSegmentation.from_pretrained(
            "sayeed99/segformer_b3_clothes"
        ).to(self.device)

        # Load CLIP model for embeddings
        logger.info("Loading CLIP model...")
        self.clip_processor = AutoProcessor.from_pretrained(
            "Marqo/marqo-fashionCLIP", trust_remote_code=True
        )
        self.clip_model = AutoModel.from_pretrained(
            "Marqo/marqo-fashionCLIP", trust_remote_code=True
        ).to(self.device)

        # Initialize Azure OpenAI client
        self.llm_client = AzureOpenAI(
            api_key=os.getenv("TIGER_AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("TIGER_AZURE_OPENAI_URL"),
            api_version="2024-08-01-preview",
        )

        # Define the mapping of class IDs to clothing item names
        self.id2label = {
            0: "Background",
            # 1: "Hat",
            # 2: "Hair",
            3: "Sunglasses",
            4: "Upper-clothes",
            5: "Skirt",
            6: "Pants",
            7: "Dress",
            8: "Belt",
            # 9: "Left-shoe",
            # 10: "Right-shoe",
            # 11: "Face",
            # 12: "Left-leg",
            # 13: "Right-leg",
            # 14: "Left-arm",
            # 15: "Right-arm",
            16: "Bag",
            17: "Scarf",
        }

        # Visualization directory
        self.visualize_dir = visualize_dir
        if self.visualize_dir and not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
            logger.info(f"Created visualization directory at {self.visalize_dir}")

    def load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from a local path or URL.
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
    ) -> Tuple[List[Dict], List[str | None]]:
        """
        Segments the given image to extract individual clothing items.

        Args:
            image (Image.Image): The input image to be segmented.
            visualize (bool, optional): If True, visualizes the segments and saves the images. Defaults to False.
            image_id (str, optional): An identifier for the image, used for saving visualizations. Defaults to "".
            single_product_mode (bool, optional): If True, only the largest segment is kept. Defaults to False.

        Returns:
            Tuple[List[Dict], List[str|None]]: A tuple containing:
                - A list of dictionaries, each containing 'label', 'image', and 'area' for each extracted item.
                - A list of file paths where the visualized segments are saved (if visualize is True).

        Raises:
            Exception: If an error occurs during segmentation.
        """

        logger.info("Processing image for segmentation...")
        try:
            inputs = self.segmentation_processor(images=image, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.segmentation_model(**inputs)
                logits = (
                    outputs.logits
                )  # Shape: [batch_size, num_classes, height, width]
                upsampled_logits = nn.functional.interpolate(
                    logits,
                    size=image.size[::-1],  # (height, width)
                    mode="bilinear",
                    align_corners=False,
                )
                seg = upsampled_logits.argmax(dim=1)[0]  # Shape: [height, width]

            logger.debug(f"Segmentation: {seg}")

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
        """
        try:
            if self.visualize_dir:
                filename = f"{image_id}_{label.replace(',', '').replace(' ', '_')}.png"
                filepath = os.path.join(self.visualize_dir, filename)
                item_image.save(filepath)
                logger.info(f"Saved visualization: {filepath}")
                return filepath
            else:
                # Display the image using matplotlib
                plt.imshow(item_image)
                plt.title(label)
                plt.axis("off")
                plt.show()
        except Exception as e:
            logger.error(f"Error visualizing segment: {e}")
        return None

    def get_embedding(self, item: Dict) -> EmbeddingData:
        """
        Generate embeddings for a clothing item using the fine-tuned fashion CLIP model.
        """
        logger.info(f"Generating embedding for item: {item['label']}")
        try:
            text_input = [f"a photo of {item['label']}"]
            image_input = [item["image"]]

            processed = self.clip_processor(
                text=text_input,
                images=image_input,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(
                    processed["pixel_values"], normalize=True
                )
                embedding = image_features.cpu().numpy()[0].tolist()
            return EmbeddingData(label=item["label"], embedding=embedding)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def encode_image(self, image: Image.Image) -> str:
        """
        Encode a PIL Image to a base64 string.
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def extract_attributes(self, item: Dict) -> ProductAttributes:
        """
        Use GPT-4o to extract attributes from an image segment.
        """
        logger.info(f"Extracting attributes for item: {item['label']}")
        try:
            # Encode the image to base64
            image_base64 = self.encode_image(item["image"])

            # Define the prompt
            prompt = [
                {
                    "role": "system",
                    "content": "You are an AI assistant that can analyze images and extract attributes from them and return the attributes in JSON format.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image and extract the following attributes, ensuring that the output JSON follows the specified format and uses only the provided options for each attribute value.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                        {"type": "text", "text": ATTRIBUTES_TO_EXTRACT_PROMPT},
                    ],
                },
            ]

            # Call GPT-4o
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=prompt,
            )

            assistant_message = response.choices[0].message.content
            logger.debug(f"GPT-4o response: {assistant_message}")

            # Parse the assistant's message as JSON
            import json

            attributes = json.loads(assistant_message)

            return ProductAttributes(label=item["label"], attributes=attributes)
        except Exception as e:
            logger.error(f"Error extracting attributes: {e}")
            # Fallback to unknown attributes
            attributes = {
                "type": "unknown",
                "color": "unknown",
                "style": ["unknown"],
                "season": ["unknown"],
                "occasion": ["unknown"],
                "price": "unknown",
                "material": ["unknown"],
                "pattern": "unknown",
                "fit": "unknown",
                "gender": "unknown",
                "age_group": "unknown",
            }
            return ProductAttributes(label=item["label"], attributes=attributes)

    def process_image(
        self,
        image_path_or_url: str,
        visualize: bool = False,
        image_id: str = "",
        single_product_mode: bool = False,
        skip_attribute_extraction: bool = False,
    ) -> Tuple[List[Dict], List[str | None]]:
        """
        Parameters:
            image_path_or_url (str): Path to the local image file or URL of the image to process
            visualize (bool, optional): If True, visualizes the segmentation results. Defaults to False.
            image_id (str, optional): Identifier for the image being processed. Defaults to "".
            single_product_mode (bool, optional): If True, processes image as single product. Defaults to False.
            skip_attribute_extraction (bool, optional): If True, skips attribute extraction step. Defaults to False.

        Returns:
            Tuple[List[Dict], List[str]]: A tuple containing:
                - List of dictionaries with processed items, each containing:
                    - label: Item classification label
                    - attributes: Extracted attributes (None if skip_attribute_extraction is True)
                    - embedding: Vector embedding of the item
                - List of file paths for any saved visualizations

        Raises:
            Exception: If there's an error during image processing
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
                if not skip_attribute_extraction:
                    attributes = self.extract_attributes(item)
                else:
                    attributes = None
                embedding = self.get_embedding(item)
                result = {
                    "label": item["label"],
                    "attributes": attributes,
                    "embedding": embedding,
                }
                results.append(result)

                logger.info(f"Processed item: {item['label']}")
                logger.debug(f"Attributes: {attributes}")
                logger.debug(f"Embedding size: {len(embedding.embedding)}")

            logger.info("Finished processing image.")
            return results, filepaths
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise


if __name__ == "__main__":
    # Example usage
    from pprint import pprint

    processor = ImageProcessor(visualize_dir="output/segmented_images")
    results, _ = processor.process_image(
        image_path_or_url="dataset/celebrity_clothes/celebrity_4_bottom.jpg",
        visualize=True,
        image_id="celebrity_4_bottom",
        single_product_mode=False,
        skip_attribute_extraction=False,
    )
    pprint(results)
