# API Documentation

This document provides comprehensive API documentation for the Complete the Look fashion recommendation system.

## Table of Contents

- [Overview](#overview)
- [Core Components](#core-components)
- [Database APIs](#database-apis)
- [Model APIs](#model-apis)
- [Inference APIs](#inference-apis)
- [Streamlit Interface](#streamlit-interface)

## Overview

The Complete the Look system provides APIs for:
- Image processing and attribute extraction
- Knowledge graph operations
- Vector similarity search
- Fashion recommendations
- Social media analysis

## Core Components

### Image Processor

The main orchestrator for image processing tasks.

```python
from src.models.model_manager import image_processor

# Process a single image
items, metadata = image_processor.process_image(
    image_path: str,
    image_id: Optional[str] = None,
    single_product_mode: bool = False,
    skip_attribute_extraction: bool = False,
    skip_style_extraction: bool = False,
    visualize: bool = False
)
```

**Parameters:**
- `image_path`: Path to the image file
- `image_id`: Optional identifier for the image
- `single_product_mode`: If True, assumes single product per image
- `skip_attribute_extraction`: Skip LLM-based attribute extraction
- `skip_style_extraction`: Skip style description generation
- `visualize`: Generate visualization of segments

**Returns:**
- `items`: List of detected clothing items with attributes and embeddings
- `metadata`: Additional processing metadata

### Graph Database Handler

Manages Neo4j graph database operations.

```python
from src.database.graph_database import GraphDatabaseHandler

# Initialize connection
graph_db = GraphDatabaseHandler(
    uri: str,
    user: str,
    password: str
)

# Create product node
graph_db.create_product_node(
    product_id: str,
    attributes: Dict[str, Any]
)

# Create relationship
graph_db.create_or_update_relationship(
    product_id1: str,
    product_id2: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None
)

# Query recommendations
recommendations = graph_db.get_recommendations(
    product_id: str,
    relationship_type: str = "WORN_WITH",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10
)
```

### Vector Database Handler

Manages Pinecone vector database operations.

```python
from src.database.vector_database import VectorDatabase

# Initialize connection
vector_db = VectorDatabase(
    api_key: str,
    host: str,
    index_name: str
)

# Upsert embeddings
vector_db.upsert_embeddings(
    embeddings: List[Dict[str, Any]],
    namespace: str = "default"
)

# Query similar items
results = vector_db.query(
    embedding: List[float],
    top_k: int = 5,
    namespace: str = "default",
    include_values: bool = False,
    filters: Optional[Dict[str, Any]] = None
)
```

## Model APIs

### Segmentation Model

Detects and segments clothing items in images.

```python
from src.models.segmentation_model import SegmentationModel

segmentation_model = SegmentationModel(model_name: str = "sayeed99/segformer_b3_clothes")

# Detect clothing items
segments = segmentation_model.detect(image_path: str)
```

### Embedding Model

Generates embeddings for images and text.

```python
from src.models.embedding_model import EmbeddingModel

embedding_model = EmbeddingModel(model_name: str = "Marqo/marqo-fashionCLIP")

# Generate image embedding
image_embedding = embedding_model.encode_image(image_path: str)

# Generate text embedding
text_embedding = embedding_model.encode_text(text: str)
```

### Attribute Extraction Model

Extracts structured attributes from images using LLM.

```python
from src.models.attribute_extraction_model import AttributeExtractionModel

attribute_model = AttributeExtractionModel()

# Extract attributes from image
attributes = attribute_model.extract_attributes(
    image_path: str,
    product_description: Optional[str] = None
)
```

## Inference APIs

### Recommender System

Main recommendation engine that combines graph and vector search.

```python
from src.inference.recommender import Recommender

# Initialize recommender
recommender = Recommender(
    graph_db: GraphDatabaseHandler,
    catalog_csv_path: str,
    vector_db_image: VectorDatabase,
    vector_db_style: VectorDatabase
)

# Get recommendations for a product
recommendations = recommender.get_recommendations(
    selected_product_id: str,
    filters: Optional[Dict[str, Any]] = None,
    threshold: int = 1,
    top_k: int = 5
)

# Find similar items by image
similar_items = recommender.find_similar_by_image(
    image_path: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
)

# Find similar items by text description
similar_items = recommender.find_similar_by_text(
    text_description: str,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
)
```

### Product Attributes

Extracts and manages product attributes.

```python
from src.inference.product_attributes import AttributeExtractionModel

attribute_extractor = AttributeExtractionModel()

# Extract attributes from unstructured image
attributes = attribute_extractor.extract_from_image(
    image_path: str,
    product_info: Optional[Dict[str, Any]] = None
)

# Generate style description
style_description = attribute_extractor.generate_style_description(
    attributes: Dict[str, Any]
)
```

## Data Models

### Product Item

```python
from src.utils.models import ProductItem

class ProductItem:
    product_id: str
    image_path: str
    label: str
    embedding: List[float]
    style_embedding: Optional[List[float]]
    attributes: Dict[str, Any]
    style_description: Optional[str]
    confidence: float
```

### Recommendation Result

```python
from src.utils.models import RecommendationResult

class RecommendationResult:
    selected_product: Dict[str, Any]
    worn_with: List[Dict[str, Any]]
    complemented: List[Dict[str, Any]]
    metadata: Dict[str, Any]
```

## Streamlit Interface

The Streamlit application provides a web interface with the following pages:

### 1. Product Attribute Extraction

```python
# Upload image and extract attributes
uploaded_file = st.file_uploader("Upload Product Image", type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    attributes = attribute_extractor.extract_from_image(uploaded_file)
    st.json(attributes)
```

### 2. Complete the Look

```python
# Select product and get recommendations
selected_product = st.selectbox("Select a Product", product_options)
if selected_product:
    recommendations = recommender.get_recommendations(selected_product)
    display_recommendations(recommendations)
```

### 3. Style Matching

```python
# Upload outfit image
uploaded_image = st.file_uploader("Upload Outfit Image", type=['png', 'jpg', 'jpeg'])
if uploaded_image:
    similar_items = recommender.find_similar_by_image(uploaded_image)
    display_similar_items(similar_items)

# Text-based search
text_query = st.text_input("Describe your style")
if text_query:
    similar_items = recommender.find_similar_by_text(text_query)
    display_similar_items(similar_items)
```

## Error Handling

All APIs include comprehensive error handling:

```python
try:
    result = api_call()
except ValueError as e:
    logger.error(f"Invalid input: {e}")
    raise
except ConnectionError as e:
    logger.error(f"Database connection failed: {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Performance Considerations

### Caching

The system implements caching for:
- Model loading and initialization
- Database connections
- Frequently accessed embeddings

### Batch Processing

For large datasets, use batch processing:

```python
# Process catalog in batches
def process_catalog_batch(catalog_df: pd.DataFrame, batch_size: int = 100):
    for i in range(0, len(catalog_df), batch_size):
        batch = catalog_df[i:i + batch_size]
        process_catalog_batch_items(batch)
```

### Async Operations

For I/O-bound operations, consider async processing:

```python
import asyncio

async def process_images_async(image_paths: List[str]):
    tasks = [process_single_image(path) for path in image_paths]
    results = await asyncio.gather(*tasks)
    return results
```

## Configuration

### Environment Variables

All configuration is managed through environment variables:

```bash
# Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Vector Database
PINECONE_API_KEY=your_api_key
PINECONE_HOST_IMAGE=your_host
PINECONE_HOST_STYLE=your_host

# Models
SEGMENTATION_MODEL=sayeed99/segformer_b3_clothes
EMBEDDING_MODEL=Marqo/marqo-fashionCLIP
TEXT_EMBEDDING_MODEL=all-MiniLM-L6-v2

# OpenAI
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
```

### Model Configuration

Models can be configured through the model manager:

```python
from src.models.model_manager import ModelManager

model_manager = ModelManager(
    segmentation_model_name: str = "sayeed99/segformer_b3_clothes",
    embedding_model_name: str = "Marqo/marqo-fashionCLIP",
    text_embedding_model_name: str = "all-MiniLM-L6-v2"
)
```

## Testing

### Unit Tests

```python
import pytest
from unittest.mock import Mock

def test_recommender_get_recommendations():
    # Arrange
    mock_graph_db = Mock()
    mock_vector_db = Mock()
    recommender = Recommender(mock_graph_db, "test.csv", mock_vector_db, mock_vector_db)
    
    # Act
    result = recommender.get_recommendations("test_product")
    
    # Assert
    assert isinstance(result, dict)
    assert "selected_product" in result
```

### Integration Tests

```python
def test_end_to_end_recommendation():
    # Test complete recommendation pipeline
    image_path = "test_image.jpg"
    items, _ = image_processor.process_image(image_path)
    assert len(items) > 0
    
    recommendations = recommender.get_recommendations(items[0]["product_id"])
    assert "worn_with" in recommendations
```

## Monitoring and Logging

### Logging Configuration

```python
from loguru import logger

# Configure logging
logger.add(
    "logs/app.log",
    rotation="1 day",
    retention="30 days",
    level="INFO"
)
```

### Performance Monitoring

```python
import time

def monitor_performance(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## Security Considerations

### API Key Management

- Store API keys in environment variables
- Never commit keys to version control
- Use secure key management services in production

### Input Validation

```python
def validate_image_path(image_path: str) -> bool:
    """Validate image path and file type."""
    if not os.path.exists(image_path):
        raise ValueError("Image file does not exist")
    
    allowed_extensions = ['.jpg', '.jpeg', '.png']
    if not any(image_path.lower().endswith(ext) for ext in allowed_extensions):
        raise ValueError("Unsupported image format")
    
    return True
```

### Rate Limiting

Implement rate limiting for external API calls:

```python
from functools import wraps
import time

def rate_limit(calls_per_second: int = 10):
    def decorator(func):
        last_call_time = 0
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal last_call_time
            current_time = time.time()
            time_since_last_call = current_time - last_call_time
            
            if time_since_last_call < 1.0 / calls_per_second:
                time.sleep(1.0 / calls_per_second - time_since_last_call)
            
            last_call_time = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

---

For more detailed information about specific components, refer to the individual module documentation and the main README file.
