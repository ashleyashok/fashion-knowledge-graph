# src/models/model_manager.py

from src.models.segmentation_model import SegmentationModel
from src.models.embedding_model import ClipEmbeddingModel, VertexAIEmbeddingModel
from src.models.attribute_extraction_model import AttributeExtractionModel
from src.engine.image_processor import ImageProcessor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define id2label mapping specific to the segmentation model
# If your model's config does not have id2label, you need to define it
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

# Initialize models
segmentation_model = SegmentationModel(
    model_name="sayeed99/segformer_b3_clothes",
    device=device,
    id2label=id2label,
)

# embedding_model = ClipEmbeddingModel(
#     model_name="Marqo/marqo-fashionCLIP",
#     device=device,
# )

embedding_model = VertexAIEmbeddingModel()

attribute_model = AttributeExtractionModel()

# Initialize ImageProcessor
image_processor = ImageProcessor(
    segmentation_model=segmentation_model,
    embedding_model=embedding_model,
    attribute_model=attribute_model,
    visualize_dir="temp_images/segmented_images",
)
