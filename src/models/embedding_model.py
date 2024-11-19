from typing import List, Optional
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
from loguru import logger


class EmbeddingModel:
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
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

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
