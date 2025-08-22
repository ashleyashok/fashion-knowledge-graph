"""
Configuration settings for the Complete the Look fashion recommendation system.

This module centralizes all application settings, environment variables,
and configuration constants used throughout the system.
"""

import os
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    pinecone_api_key: str
    pinecone_host_image: str
    pinecone_host_style: str

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create DatabaseConfig from environment variables."""
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_host_image=os.getenv("PINECONE_HOST_IMAGE", ""),
            pinecone_host_style=os.getenv("PINECONE_HOST_STYLE", ""),
        )


@dataclass
class ModelConfig:
    """Configuration for ML models."""

    segmentation_model: str
    embedding_model: str
    text_embedding_model: str
    device: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        """Create ModelConfig from environment variables."""
        import torch

        return cls(
            segmentation_model=os.getenv(
                "SEGMENTATION_MODEL", "sayeed99/segformer_b3_clothes"
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "Marqo/marqo-fashionCLIP"),
            text_embedding_model=os.getenv("TEXT_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI/Azure OpenAI services."""

    api_key: str
    endpoint: str
    api_version: str

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create OpenAIConfig from environment variables."""
        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        )


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""

    catalog_csv_path: str
    temp_images_dir: str
    segmented_images_dir: str
    user_uploaded_dir: str

    @classmethod
    def from_env(cls) -> "PathConfig":
        """Create PathConfig with default paths."""
        base_dir = Path(__file__).parent.parent.parent

        return cls(
            catalog_csv_path=str(base_dir / "output" / "data" / "catalog_combined.csv"),
            temp_images_dir=str(base_dir / "temp_images"),
            segmented_images_dir=str(base_dir / "temp_images" / "segmented_images"),
            user_uploaded_dir=str(base_dir / "temp_images" / "user_uploaded"),
        )


@dataclass
class AppConfig:
    """Main application configuration."""

    database: DatabaseConfig
    models: ModelConfig
    openai: OpenAIConfig
    paths: PathConfig

    # Segmentation model label mapping
    segmentation_labels: Dict[int, str] = None

    def __post_init__(self):
        """Initialize default values after object creation."""
        if self.segmentation_labels is None:
            self.segmentation_labels = {
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

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create AppConfig from environment variables."""
        return cls(
            database=DatabaseConfig.from_env(),
            models=ModelConfig.from_env(),
            openai=OpenAIConfig.from_env(),
            paths=PathConfig.from_env(),
        )

    def validate(self) -> None:
        """Validate that all required configuration is present."""
        missing_vars = []

        if not self.database.neo4j_password:
            missing_vars.append("NEO4J_PASSWORD")
        if not self.database.pinecone_api_key:
            missing_vars.append("PINECONE_API_KEY")
        if not self.database.pinecone_host_image:
            missing_vars.append("PINECONE_HOST_IMAGE")
        if not self.database.pinecone_host_style:
            missing_vars.append("PINECONE_HOST_STYLE")
        if not self.openai.api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not self.openai.endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )


# Global configuration instance
config = AppConfig.from_env()
