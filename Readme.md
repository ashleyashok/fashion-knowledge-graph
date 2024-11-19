# Complete the Look - Outfit Recommendation System

This application leverages advanced image processing, knowledge graphs, and recommendation systems to provide personalized outfit recommendations. By analyzing images from social media or user uploads, the system identifies clothing items and suggests complementary products from a retailer's catalog.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Processing the Catalog](#processing-the-catalog)
  - [Processing Social Media Images](#processing-social-media-images)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The **Complete the Look** project aims to enhance the online shopping experience by:

- Extracting clothing items from images using computer vision.
- Building a knowledge graph to understand relationships between products.
- Using vector databases for efficient similarity search.
- Providing a Streamlit dashboard for interactive recommendations.

This concept can be applied to various industries beyond fashion retail, such as interior design, automotive customization, beauty and cosmetics, and more.

## Features

- **Image Processing**: Segment and identify clothing items in images using advanced computer vision models.
- **Attribute Extraction**: Extract attributes like type, color, and style using language models.
- **Knowledge Graph**: Build relationships between products based on co-occurrence in images.
- **Vector Database**: Store embeddings for fast similarity searches.
- **Streamlit Dashboard**: Interactive web application for users to upload images or select products and receive recommendations.
- **Modular Design**: Easily swap out models for segmentation, embedding, and attribute extraction.
- **Flexible Input Methods**: Accept both image uploads and image URLs for processing.

## Architecture Overview

The system consists of several interconnected components:

1. **Image Processor**: Uses modular models to segment images, extract embeddings, and attributes for each clothing item.
2. **Graph Database**: Stores products and relationships using Neo4j to represent the knowledge graph.
3. **Vector Database**: Stores item embeddings for efficient similarity queries using vector search (e.g., Pinecone).
4. **Recommender System**: Combines data from the graph and vector databases to generate recommendations.
5. **Streamlit App**: Provides a user interface for interacting with the system.

## Installation

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- Neo4j database instance
- Access to a vector database service (e.g., Pinecone)
- GPU (recommended for faster image processing)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/complete-the-look.git
   cd complete-the-look
   ```

2. **Install Dependencies**

   Use Poetry to install all required packages.

   ```bash
   poetry install
   ```

3. **Set Up Environment Variables**

   Create a `.env` file in the project root directory with the following variables:

   ```env
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_neo4j_username
   NEO4J_PASSWORD=your_neo4j_password
   PINECONE_API_KEY=your_pinecone_api_key
   TIGER_AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   TIGER_AZURE_OPENAI_URL=your_azure_openai_endpoint
   ```

4. **Prepare the Catalog Data**

   Place your retailer's catalog CSV file in the `output/data/` directory and ensure it's named `catalog_combined.csv`. The CSV should contain at least the following columns:

   - `product_id`
   - `image_path`
   - `category` (or `type`)

5. **Initialize the Vector Database**

   Create an index in your vector database for storing embeddings. For example, using Pinecone:

   ```python
   import pinecone

   pinecone.init(api_key='your_pinecone_api_key', environment='your_environment')
   pinecone.create_index('catalog-clothes', dimension=512)  # Adjust dimension as per your embedding model
   ```

6. **Process the Catalog**

   Run the script to process the catalog and populate the databases:

   ```bash
   python src/engine/process_catalog.py
   ```

## Usage

### Processing the Catalog

Before running the application, you need to process your product catalog to generate embeddings and populate the vector and graph databases.

```bash
python src/engine/process_catalog.py
```

### Processing Social Media Images

(Optional) You can process social media images to enrich the knowledge graph with real-world outfit combinations.

```bash
python src/engine/process_social_media_images.py
```

### Running the Streamlit App

Start the Streamlit application to interact with the recommender system.

```bash
streamlit run app/main.py
```

**Features of the Streamlit App:**

- **Product Recommendations**

  - Select a product from the catalog to get recommendations.
  - Apply attribute-based filters to refine recommendations.
  - View products that are often worn with or complement the selected item.

- **Style Match: Upload Your Outfit**

  - Upload an image or enter an image URL.
  - The system analyzes the outfit and finds matching products from the catalog.
  - Displays products that closely match the items in the uploaded image.

## Project Structure

```
complete-the-look/
├── app/
│   └── main.py                   # Streamlit application
├── src/
│   ├── engine/
│   │   ├── image_processor.py    # Orchestrates image processing tasks
│   │   ├── process_catalog.py    # Processes the catalog data
│   │   └── process_social_media_images.py  # Processes social media images
│   ├── inference/
│   │   └── recommender.py        # Recommender system logic
│   ├── models/
│   │   ├── model_manager.py      # Centralized model initialization
│   │   ├── segmentation_model.py # Segmentation model class
│   │   ├── embedding_model.py    # Embedding model class
│   │   └── attribute_extraction_model.py  # Attribute extraction model class
│   ├── database/
│   │   ├── graph_database.py     # Graph database handler
│   │   └── vector_database.py    # Vector database handler
│   └── utils/
│       ├── models.py             # Data models and schemas
│       └── prompts.py            # Prompts for LLMs
├── output/
│   └── data/
│       └── catalog_combined.csv  # Catalog data file
├── temp_images/                  # Temporary images directory
├── README.md                     # Project documentation
└── .env                          # Environment variables
```

## Dependencies

- **Python Libraries**

  - **Streamlit**: For building the web application interface.
  - **Pandas**: Data manipulation and analysis.
  - **NumPy**: Numerical operations.
  - **Pillow**: Image processing.
  - **Requests**: Handling HTTP requests (for image URLs).
  - **Loguru**: Advanced logging.
  - **Transformers**: Pretrained models for NLP and computer vision.
  - **Torch**: Deep learning library for model computations.
  - **Matplotlib**: Visualization of segmented images.
  - **OpenAI**: For interacting with Azure OpenAI services.

- **Databases**

  - **Neo4j**: Graph database for storing products and relationships.
  - **Vector Database**: For storing embeddings (e.g., Pinecone).

- **Models**

  - **Segmentation Model**: Pretrained model for image segmentation (e.g., `sayeed99/segformer_b3_clothes`).
  - **Embedding Model**: Pretrained model for generating embeddings (e.g., `Marqo/marqo-fashionCLIP`).
  - **Attribute Extraction Model**: Uses Azure OpenAI's GPT-4o for extracting attributes.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes with clear messages.
4. Open a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Note**: This project uses advanced machine learning models, including computer vision and natural language processing. Ensure you have the necessary computational resources and permissions to use these models.

**Disclaimer**: The performance of the recommendation system depends on the quality of the models and data used. Always test thoroughly before deploying in a production environment.