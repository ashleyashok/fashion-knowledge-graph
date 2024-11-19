# src/models/segmentation_model.py

from typing import Dict, Optional
import torch
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import torch.nn.functional as F
from loguru import logger


class SegmentationModel:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the SegmentationModel.

        Parameters
        ----------
        model_name : str
            The name or path of the segmentation model to load.
        device : str, optional
            The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        id2label : dict, optional
            Mapping from class IDs to labels. If not provided, attempts to load from model config.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing SegmentationModel on device: {self.device}")
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(self.device)

        # Define id2label mapping
        if id2label is not None:
            self.id2label = id2label
            logger.info("Using provided id2label mapping.")
        elif hasattr(self.model.config, 'id2label') and self.model.config.id2label:
            self.id2label = self.model.config.id2label
            logger.info("Loaded id2label mapping from model config.")
        else:
            logger.warning("No id2label mapping found. Please provide id2label when initializing SegmentationModel.")
            self.id2label = {}

    def get_id2label(self) -> Dict[int, str]:
        """
        Get the id2label mapping.

        Returns
        -------
        Dict[int, str]
            Mapping from class IDs to labels.
        """
        return self.id2label

    def segment(self, image: Image.Image) -> torch.Tensor:
        """
        Perform segmentation on the input image.

        Parameters
        ----------
        image : PIL.Image.Image
            The input image.

        Returns
        -------
        torch.Tensor
            The segmentation map as a tensor.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [batch_size, num_classes, height, width]
            upsampled_logits = F.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False,
            )
            segmentation = upsampled_logits.argmax(dim=1)[0]  # [height, width]
        return segmentation
