# src/models/embedding_model.py

from abc import ABC, abstractmethod
from typing import List, Literal, Optional

import PIL
import torch
import vertexai
from loguru import logger
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoProcessor
from vertexai.vision_models import Image as VertexImage
from vertexai.vision_models import MultiModalEmbeddingModel


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def get_embedding(
        self,
        image: Optional[Image.Image],
        text: Optional[str],
        type: Literal["image", "text"] = "image",
    ) -> List[float]:
        pass


class ClipEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the EmbeddingModel.

        Parameters
        ----------
        model_name : str
            The name or path of the embedding model to load.
        device : str, optional
            The device to run the model on ('cuda' or 'cpu'). Defaults to 'cuda' if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing EmbeddingModel on device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            self.device
        )

    def get_embedding(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        type: Literal["image", "text"] = "image",
    ) -> List[float]:
        """
        Generate an embedding for the given image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to embed.
         text : str
            The label or description of the image.
        type: str
            The type of embedding to generate, either "image" or "text".

        Returns
        -------
        List[float]
            The image or text embedding vector.
        """
        try:
            text_input = [f"a photo of {text} clothing"]
            if not image:
                image = PIL.Image.open("dataset/macy_clothes/macy_1.jpeg")
            image_input = [image]

            processed = self.processor(
                text=text_input,
                images=image_input,
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                if type == "image":
                    image_features = self.model.get_image_features(
                        processed["pixel_values"], normalize=True
                    )
                    embedding = image_features.cpu().numpy()[0].tolist()
                elif type == "text":
                    text_features = self.model.get_text_features(
                        processed["input_ids"], normalize=True
                    )
                    embedding = text_features.cpu().numpy()[0].tolist()

            return embedding

        except Exception as e:
            logger.error(f"Error generating image embedding: {e}")
            raise RuntimeError from e


class VertexAIEmbeddingModel(BaseEmbeddingModel):
    """
    A class to interact with Vertex AI's MultiModalEmbeddingModel to generate image
    or text embeddings.

    Attributes:
        model (MultiModalEmbeddingModel): The pre-trained multimodal embedding model.

    """

    def __init__(
        self, project_id: str = "gemini-copilot-testing", location: str = "us-central1"
    ):
        vertexai.init(project=project_id, location=location)
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    def get_embedding(
        self,
        image: Optional[Image.Image],
        text: Optional[str],
        type: Literal["image", "text"] = "image",
    ) -> List[float]:
        """
        Generate an embedding for the given image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to embed.
        text : str
            The label or description of the image.
        type: str
            The type of embedding to generate, either "image" or "text".

        Returns
        -------
        List[float]
            The image or text embedding vector.
        """
        try:
            # Convert PIL Image to bytes
            import io

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()

            # Create Vertex AI Image object
            vertex_image = VertexImage(image_bytes=img_byte_arr)

            # Get embeddings
            embeddings = self.model.get_embeddings(
                image=vertex_image,
                contextual_text=f"a photo of {text} clothing",
                dimension=1408,
            )
            return (
                embeddings.image_embedding
                if type == "image"
                else embeddings.text_embedding
            )
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding model using Sentence-Transformer for text embeddings.
    """

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None
    ):
        """
        Initialize the SentenceTransformerEmbeddingModel.

        Parameters
        ----------
        model_name : str, optional
            The name or path of the Sentence-Transformer model to load.
            Default is 'all-MiniLM-L6-v2'.
        device : str, optional
            The device to run the model on ('cuda' or 'cpu').
            Defaults to 'cuda' if available.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(
            f"Initializing SentenceTransformerEmbeddingModel on device: {self.device}"
        )
        self.model = SentenceTransformer(model_name, device=self.device)

    def get_embedding(
        self,
        image: Optional[Image.Image] = None,
        text: Optional[str] = None,
        type: Literal["image", "text"] = "text",
    ) -> List[float]:
        """
        Generate an embedding for the given text.

        Parameters
        ----------
        image : PIL.Image.Image, optional
            Not used in this model.
        text : str
            The text to embed.
        type : str
            Must be 'text' for this model.

        Returns
        -------
        List[float]
            The text embedding vector.
        """
        if type != "text":
            raise ValueError(
                "SentenceTransformerEmbeddingModel only supports text embeddings."
            )
        if text is None:
            raise ValueError("Text input must be provided for text embeddings.")
        try:
            embedding = self.model.encode(text).tolist()
            return embedding
        except Exception as e:
            logger.error(f"Error generating text embedding: {e}")
            raise


if __name__ == "__main__":
    # Example usage of the embedding models
    st_model = SentenceTransformerEmbeddingModel()

    # Get embeddings
    sentence = 'The style embodies a "Barbie theme" with playful, vibrant designs, featuring bold silhouettes, spar'
    text_embedding = st_model.get_embedding(text=sentence)
    print(text_embedding)
