"""
Model manager for the Complete the Look fashion recommendation system.

This module provides centralized model management, initialization,
and configuration for all machine learning models used in the system.
"""

from typing import Optional
from loguru import logger

from src.config.settings import config
from src.models.segmentation_model import SegmentationModel
from src.models.embedding_model import (
    ClipEmbeddingModel,
    VertexAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from src.models.attribute_extraction_model import AttributeExtractionModel
from src.engine.image_processor import ImageProcessor


class ModelManager:
    """
    Centralized manager for all machine learning models.

    This class handles the initialization, configuration, and management
    of all models used in the fashion recommendation system, including
    segmentation, embedding, and attribute extraction models.
    """

    def __init__(self, config_instance: Optional[object] = None):
        """
        Initialize the model manager.

        Parameters
        ----------
        config_instance : object, optional
            Configuration object. If None, uses the global config.
        """
        self.config = config_instance or config
        self._models = {}
        self._image_processor = None

        logger.info("Initializing ModelManager")

    def initialize_models(self) -> None:
        """
        Initialize all models used in the system.

        This method creates and configures all required models:
        - Segmentation model for clothing detection
        - CLIP embedding model for image/text embeddings
        - Sentence Transformer for text embeddings
        - Attribute extraction model for product attributes
        - Image processor for orchestration
        """
        try:
            logger.info("Initializing all models...")

            # Initialize segmentation model
            self._models["segmentation"] = SegmentationModel(
                model_name=self.config.models.segmentation_model,
                device=self.config.models.device,
                id2label=self.config.segmentation_labels,
            )

            # Initialize CLIP embedding model
            self._models["embedding"] = ClipEmbeddingModel(
                model_name=self.config.models.embedding_model,
                device=self.config.models.device,
            )

            # Initialize text embedding model
            self._models["text_embedding"] = SentenceTransformerEmbeddingModel(
                model_name=self.config.models.text_embedding_model,
                device=self.config.models.device,
            )

            # Initialize attribute extraction model
            self._models["attribute"] = AttributeExtractionModel()

            # Initialize image processor
            self._image_processor = ImageProcessor(
                segmentation_model=self._models["segmentation"],
                embedding_model=self._models["embedding"],
                text_embedding_model=self._models["text_embedding"],
                attribute_model=self._models["attribute"],
                visualize_dir=self.config.paths.segmented_images_dir,
            )

            logger.info("All models initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def load_all_models(self) -> None:
        """
        Load all models into memory.

        This method ensures all models are loaded and ready for inference.
        Models are loaded lazily to avoid unnecessary memory usage.
        """
        try:
            logger.info("Loading all models into memory...")

            for name, model in self._models.items():
                logger.info(f"Loading {name} model...")
                model.ensure_model_loaded()

            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def get_model(self, model_name: str):
        """
        Get a specific model by name.

        Parameters
        ----------
        model_name : str
            Name of the model to retrieve.

        Returns
        -------
        object
            The requested model instance.

        Raises
        ------
        KeyError
            If the model name is not found.
        """
        if model_name not in self._models:
            available_models = list(self._models.keys())
            raise KeyError(
                f"Model '{model_name}' not found. Available models: {available_models}"
            )
        return self._models[model_name]

    @property
    def segmentation_model(self):
        """Get the segmentation model."""
        return self.get_model("segmentation")

    @property
    def embedding_model(self):
        """Get the CLIP embedding model."""
        return self.get_model("embedding")

    @property
    def text_embedding_model(self):
        """Get the text embedding model."""
        return self.get_model("text_embedding")

    @property
    def attribute_model(self):
        """Get the attribute extraction model."""
        return self.get_model("attribute")

    @property
    def image_processor(self):
        """Get the image processor."""
        if self._image_processor is None:
            raise RuntimeError(
                "Image processor not initialized. Call initialize_models() first."
            )
        return self._image_processor

    def get_model_info(self) -> dict:
        """
        Get information about all models.

        Returns
        -------
        dict
            Dictionary containing model information including names,
            devices, and loading status.
        """
        info = {}
        for name, model in self._models.items():
            info[name] = {
                "model_name": getattr(model, "model_name", "Unknown"),
                "device": getattr(model, "device", "Unknown"),
                "is_loaded": getattr(model, "_is_loaded", False),
                "embedding_dim": getattr(model, "embedding_dim", None),
            }
        return info

    def cleanup(self) -> None:
        """
        Clean up model resources.

        This method should be called when shutting down the application
        to free up memory and resources.
        """
        try:
            logger.info("Cleaning up model resources...")

            # Clear model references
            for name in list(self._models.keys()):
                del self._models[name]

            self._image_processor = None

            logger.info("Model cleanup completed")

        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")

    def __enter__(self):
        """Context manager entry."""
        self.initialize_models()
        self.load_all_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Global model manager instance
model_manager = ModelManager()


# Convenience accessors for backward compatibility
def get_segmentation_model():
    """Get the segmentation model."""
    return model_manager.segmentation_model


def get_embedding_model():
    """Get the CLIP embedding model."""
    return model_manager.embedding_model


def get_text_embedding_model():
    """Get the text embedding model."""
    return model_manager.text_embedding_model


def get_attribute_model():
    """Get the attribute extraction model."""
    return model_manager.attribute_model


def get_image_processor():
    """Get the image processor."""
    return model_manager.image_processor


# Initialize models for backward compatibility
try:
    model_manager.initialize_models()
    model_manager.load_all_models()

    # Create convenience variables
    segmentation_model = model_manager.segmentation_model
    embedding_model = model_manager.embedding_model
    text_embedding_model = model_manager.text_embedding_model
    attribute_model = model_manager.attribute_model
    image_processor = model_manager.image_processor

except Exception as e:
    logger.warning(f"Failed to initialize models during import: {e}")
    logger.info("Models will be initialized when first accessed")
