from abc import ABC, abstractmethod
from typing import List, Optional
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from loguru import logger
import vertexai
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def get_image_embedding(self, image: Image.Image, label: str) -> List[float]:
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

    def get_image_embedding(self, image: Image.Image, label: str) -> List[float]:
        """
        Generate an embedding for the given image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to embed.
        label : str
            The label or description of the image.

        Returns
        -------
        List[float]
            The image embedding vector.
        """
        text_input = [f"a photo of {label}"]
        image_input = [image]

        processed = self.processor(
            text=text_input,
            images=image_input,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            image_features = self.model.get_image_features(
                processed["pixel_values"], normalize=True
            )
            embedding = image_features.cpu().numpy()[0].tolist()
        return embedding


class VertexAIEmbeddingModel(BaseEmbeddingModel):
    """
    A class to interact with Vertex AI's MultiModalEmbeddingModel to generate image embeddings.

    Attributes:
        model (MultiModalEmbeddingModel): The pre-trained multimodal embedding model.

    Methods:
        __init__(project_id: str, location: str):
            Initializes the Vertex AI environment and loads the pre-trained model.
        
        get_image_embedding(image: Image.Image, label: str) -> List[float]:
            Generates and returns the image embedding for a given image and label.
    """
    def __init__(
        self, project_id: str = "gemini-copilot-testing", location: str = "us-central1"
    ):
        vertexai.init(project=project_id, location=location)
        self.model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

    def get_image_embedding(self, image: Image.Image, label: str) -> List[float]:
        """
        Generate an embedding for the given image.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to embed.
        label : str
            The label or description of the image.

        Returns
        -------
        List[float]
            The image embedding vector.
        """
        try:
            # Convert PIL Image to bytes
            import io
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Create Vertex AI Image object
            vertex_image = VertexImage(image_bytes=img_byte_arr)

            # Get embeddings
            embeddings = self.model.get_embeddings(
                image=vertex_image,
                contextual_text=f"a photo of {label}",
                dimension=1408,
            )
            return embeddings.image_embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise