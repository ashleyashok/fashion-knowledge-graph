"""
Embedding models for generating vector representations.

This module provides various embedding model implementations for
generating vector representations from text and images.
"""

from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from loguru import logger

from src.models.base_model import BaseEmbeddingModel


class ClipEmbeddingModel(BaseEmbeddingModel):
    """
    CLIP-based embedding model for fashion images and text.

    This class uses a pre-trained CLIP model specifically fine-tuned
    for fashion applications to generate embeddings from both images
    and text descriptions.
    """

    def __init__(
        self, model_name: str = "Marqo/marqo-fashionCLIP", device: Optional[str] = None
    ):
        """
        Initialize the CLIP embedding model.

        Parameters
        ----------
        model_name : str, default="Marqo/marqo-fashionCLIP"
            Name or path of the pre-trained CLIP model.
        device : str, optional
            Device to run the model on ('cpu', 'cuda', etc.).
        """
        super().__init__(model_name, device)
        self.processor = None

        logger.info(f"Initializing CLIP embedding model: {model_name}")

    def load_model(self) -> None:
        """
        Load the pre-trained CLIP model and processor.

        This method loads both the model and the processor required
        for preprocessing inputs.
        """
        try:
            logger.info(f"Loading CLIP model: {self.model_name}")

            # Load the model and processor
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()

            # Get embedding dimension
            self.embedding_dim = self.model.config.projection_dim

            self._is_loaded = True
            logger.info(
                f"CLIP model loaded successfully. Embedding dim: {self.embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str]] = None,
        type: str = "text",
    ) -> List[float]:
        """
        Generate embeddings for text or image input.

        Parameters
        ----------
        text : str, optional
            Text input for embedding generation.
        image : PIL.Image.Image or str, optional
            Image input for embedding generation. Can be a PIL Image
            or a path to an image file.
        type : str, default="text"
            Type of embedding to generate ('text' or 'image').

        Returns
        -------
        List[float]
            Vector embedding as a list of floats.

        Raises
        ------
        ValueError
            If the model is not loaded or invalid input is provided.
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before generating embeddings")

        if type == "text":
            if not text:
                raise ValueError("Text input is required for text embeddings")
            return self._get_text_embedding(text)
        elif type == "image":
            if not image:
                raise ValueError("Image input is required for image embeddings")
            return self._get_image_embedding(image)
        else:
            raise ValueError(f"Invalid embedding type: {type}")

    def _get_text_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding.

        Parameters
        ----------
        text : str
            Input text for embedding generation.

        Returns
        -------
        List[float]
            Text embedding vector.
        """
        try:
            # Preprocess text
            processed = self.processor(
                text=[text], padding="max_length", return_tensors="pt"
            )

            # Move to device
            processed = {k: v.to(self.device) for k, v in processed.items()}

            # Generate embedding
            with torch.no_grad():
                text_features = self.model.get_text_features(
                    processed["input_ids"], normalize=True
                )
                embedding = text_features.cpu().numpy()[0].tolist()

            return embedding

        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise

    def _get_image_embedding(self, image: Union[Image.Image, str]) -> List[float]:
        """
        Generate image embedding.

        Parameters
        ----------
        image : PIL.Image.Image or str
            Input image or path to image file.

        Returns
        -------
        List[float]
            Image embedding vector.
        """
        try:
            # Load image if path is provided
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")

            # Preprocess image
            processed = self.processor(
                images=[image], padding="max_length", return_tensors="pt"
            )

            # Move to device
            processed = {k: v.to(self.device) for k, v in processed.items()}

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(
                    processed["pixel_values"], normalize=True
                )
                embedding = image_features.cpu().numpy()[0].tolist()

            return embedding

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Sentence Transformer embedding model for text.

    This class uses a pre-trained Sentence Transformer model to generate
    high-quality text embeddings optimized for semantic similarity.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """
        Initialize the Sentence Transformer embedding model.

        Parameters
        ----------
        model_name : str, default="all-MiniLM-L6-v2"
            Name or path of the pre-trained Sentence Transformer model.
        device : str, optional
            Device to run the model on ('cpu', 'cuda', etc.).
        """
        super().__init__(model_name, device)

        logger.info(f"Initializing Sentence Transformer model: {model_name}")

    def load_model(self) -> None:
        """
        Load the pre-trained Sentence Transformer model.
        """
        try:
            logger.info(f"Loading Sentence Transformer model: {self.model_name}")

            # Load the model
            self.model = SentenceTransformer(self.model_name, device=self.device)

            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            self._is_loaded = True
            logger.info(
                f"Sentence Transformer model loaded successfully. Embedding dim: {self.embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model: {e}")
            raise

    def get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        type: str = "text",
    ) -> List[float]:
        """
        Generate text embedding.

        Parameters
        ----------
        text : str, optional
            Text input for embedding generation.
        image : Any, optional
            Not used for this model (text-only).
        type : str, default="text"
            Type of embedding (only 'text' is supported).

        Returns
        -------
        List[float]
            Text embedding vector.

        Raises
        ------
        ValueError
            If the model is not loaded or invalid input is provided.
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before generating embeddings")

        if type != "text":
            raise ValueError("Sentence Transformer only supports text embeddings")

        if not text:
            raise ValueError("Text input is required")

        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise


class VertexAIEmbeddingModel(BaseEmbeddingModel):
    """
    Google Vertex AI embedding model for multimodal inputs.

    This class uses Google's Vertex AI multimodal embedding model
    to generate embeddings from both text and images.
    """

    def __init__(
        self, model_name: str = "multimodalembedding@001", device: Optional[str] = None
    ):
        """
        Initialize the Vertex AI embedding model.

        Parameters
        ----------
        model_name : str, default="multimodalembedding@001"
            Name of the Vertex AI model to use.
        device : str, optional
            Not used for Vertex AI models (cloud-based).
        """
        super().__init__(model_name, device)

        logger.info(f"Initializing Vertex AI embedding model: {model_name}")

    def load_model(self) -> None:
        """
        Initialize Vertex AI client and model.
        """
        try:
            import vertexai
            from vertexai.vision_models import MultiModalEmbeddingModel

            logger.info(f"Loading Vertex AI model: {self.model_name}")

            # Initialize Vertex AI
            vertexai.init(project="gemini-copilot-testing", location="us-central1")

            # Load the model
            self.model = MultiModalEmbeddingModel.from_pretrained(self.model_name)

            # Set embedding dimension
            self.embedding_dim = 1408

            self._is_loaded = True
            logger.info(
                f"Vertex AI model loaded successfully. Embedding dim: {self.embedding_dim}"
            )

        except ImportError:
            logger.error(
                "Vertex AI not available. Please install google-cloud-aiplatform"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load Vertex AI model: {e}")
            raise

    def get_embedding(
        self,
        text: Optional[str] = None,
        image: Optional[Union[Image.Image, str]] = None,
        type: str = "text",
    ) -> List[float]:
        """
        Generate embeddings using Vertex AI.

        Parameters
        ----------
        text : str, optional
            Text input for embedding generation.
        image : PIL.Image.Image or str, optional
            Image input for embedding generation.
        type : str, default="text"
            Type of embedding to generate ('text' or 'image').

        Returns
        -------
        List[float]
            Embedding vector.
        """
        if not self._is_loaded:
            raise ValueError("Model must be loaded before generating embeddings")

        try:
            from vertexai.vision_models import Image as VertexImage

            if type == "text" and text:
                # Generate text embedding
                embeddings = self.model.get_embeddings(
                    contextual_text=text, dimension=self.embedding_dim
                )
                return embeddings.text_embedding

            elif type == "image" and image:
                # Load image if path is provided
                if isinstance(image, str):
                    vertex_image = VertexImage.load_from_file(image)
                else:
                    # Save PIL image temporarily and load
                    import tempfile

                    with tempfile.NamedTemporaryFile(
                        suffix=".jpg", delete=False
                    ) as tmp:
                        image.save(tmp.name, "JPEG")
                        vertex_image = VertexImage.load_from_file(tmp.name)

                # Generate image embedding
                embeddings = self.model.get_embeddings(
                    image=vertex_image, dimension=self.embedding_dim
                )
                return embeddings.image_embedding

            else:
                raise ValueError("Invalid input for embedding generation")

        except Exception as e:
            logger.error(f"Error generating Vertex AI embedding: {e}")
            raise
