"""
Segmentation model for clothing item detection.

This module provides the SegmentationModel class that uses a pre-trained
SegFormer model to segment clothing items from images.
"""

from typing import Dict, Optional
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from loguru import logger

from src.models.base_model import BaseSegmentationModel


class SegmentationModel(BaseSegmentationModel):
    """
    A segmentation model for detecting clothing items in images.

    This class uses a pre-trained SegFormer model to perform semantic
    segmentation on fashion images, identifying different types of
    clothing items such as shirts, pants, dresses, etc.
    """

    def __init__(
        self,
        model_name: str = "sayeed99/segformer_b3_clothes",
        device: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the segmentation model.

        Parameters
        ----------
        model_name : str, default="sayeed99/segformer_b3_clothes"
            Name or path of the pre-trained SegFormer model.
        device : str, optional
            Device to run the model on ('cpu', 'cuda', etc.).
        id2label : Dict[int, str], optional
            Mapping from class IDs to label names. If None, uses default
            clothing segmentation labels.
        """
        super().__init__(model_name, device, id2label)
        self.processor = None

        logger.info(f"Initializing SegmentationModel with {model_name}")

    def load_model(self) -> None:
        """
        Load the pre-trained SegFormer model and processor.

        This method loads both the model and the image processor
        required for preprocessing input images.
        """
        try:
            logger.info(f"Loading segmentation model: {self.model_name}")

            # Load the processor for image preprocessing
            self.processor = SegformerImageProcessor.from_pretrained(self.model_name)

            # Load the model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_name,
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            logger.info("Segmentation model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            raise

    def segment(self, image: Image.Image) -> torch.Tensor:
        """
        Perform semantic segmentation on an image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image to segment. Should be in RGB format.

        Returns
        -------
        torch.Tensor
            Segmentation mask tensor with shape (H, W) where each pixel
            contains the class ID of the detected clothing item.

        Raises
        ------
        ValueError
            If the model is not loaded or the image is invalid.
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before segmentation")

        if image is None:
            raise ValueError("Input image cannot be None")

        try:
            # Preprocess the image
            inputs = self.processor(images=image, return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get the predicted segmentation mask
            predicted_mask = torch.argmax(logits, dim=1)

            # Remove batch dimension and return
            return predicted_mask.squeeze(0)

        except Exception as e:
            logger.error(f"Error during segmentation: {e}")
            raise

    def segment_batch(self, images: list[Image.Image]) -> list[torch.Tensor]:
        """
        Perform segmentation on a batch of images.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            List of input images to segment.

        Returns
        -------
        list[torch.Tensor]
            List of segmentation mask tensors.
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before segmentation")

        if not images:
            return []

        try:
            # Preprocess all images
            inputs = self.processor(images=images, return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Perform batch inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

            # Get predicted masks for each image
            predicted_masks = torch.argmax(logits, dim=1)

            # Return list of individual masks
            return [mask for mask in predicted_masks]

        except Exception as e:
            logger.error(f"Error during batch segmentation: {e}")
            raise

    def get_segmentation_info(self, mask: torch.Tensor) -> Dict[str, any]:
        """
        Extract information from a segmentation mask.

        Parameters
        ----------
        mask : torch.Tensor
            Segmentation mask tensor.

        Returns
        -------
        Dict[str, any]
            Dictionary containing:
            - 'unique_classes': List of unique class IDs in the mask
            - 'class_counts': Dictionary mapping class IDs to pixel counts
            - 'class_labels': Dictionary mapping class IDs to label names
        """
        mask_np = mask.cpu().numpy()
        unique_classes = np.unique(mask_np)

        class_counts = {}
        class_labels = {}

        for class_id in unique_classes:
            if class_id in self.id2label:
                count = np.sum(mask_np == class_id)
                class_counts[class_id] = int(count)
                class_labels[class_id] = self.id2label[class_id]

        return {
            "unique_classes": unique_classes.tolist(),
            "class_counts": class_counts,
            "class_labels": class_labels,
        }

    def filter_by_area(self, mask: torch.Tensor, min_area: int = 1028) -> torch.Tensor:
        """
        Filter segmentation mask to keep only regions above minimum area.

        Parameters
        ----------
        mask : torch.Tensor
            Input segmentation mask.
        min_area : int, default=1028
            Minimum area (in pixels) for a region to be kept.

        Returns
        -------
        torch.Tensor
            Filtered segmentation mask where small regions are set to background (0).
        """
        mask_np = mask.cpu().numpy()
        filtered_mask = np.zeros_like(mask_np)

        for class_id in self.id2label.keys():
            if class_id == 0:  # Skip background
                continue

            class_mask = mask_np == class_id
            area = np.sum(class_mask)

            if area >= min_area:
                filtered_mask[class_mask] = class_id

        return torch.from_numpy(filtered_mask)
