from pydantic import BaseModel
from typing import List, Dict, Optional

class ImageSegment(BaseModel):
    label: str
    image_data: bytes  # Serialized image data

class ProductAttributes(BaseModel):
    label: Optional[str]
    attributes: Optional[Dict[str, str | List[str]]]

class EmbeddingData(BaseModel):
    label: str
    embedding: List[float]

class PreprocessingOutput(BaseModel):
    product_id: str
    attributes: ProductAttributes
    embedding: EmbeddingData

class RecommendationRequest(BaseModel):
    product_id: str

class RecommendationResponse(BaseModel):
    recommended_product_ids: List[str]
