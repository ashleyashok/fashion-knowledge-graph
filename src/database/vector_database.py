"""
Pinecone vector database handler for fashion recommendation system.

This module provides the VectorDatabase class for managing vector
embeddings in Pinecone for similarity search and retrieval.
"""

from typing import List, Dict, Optional, Any
import os
from pinecone import Pinecone
from loguru import logger


class VectorDatabase:
    """
    Handler for Pinecone vector database operations.

    This class manages all interactions with the Pinecone vector database,
    including upserting embeddings, querying for similar vectors, and
    managing namespaces.
    """

    def __init__(
        self,
        host: str,
        index_name: str = "catalog-clothes",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the vector database handler.

        Parameters
        ----------
        host : str
            The host address for the Pinecone service.
        index_name : str, default="catalog-clothes"
            The name of the index to use.
        api_key : str, optional
            The API key for authentication. If not provided, will be
            fetched from the environment variable 'PINECONE_API_KEY'.

        Raises
        ------
        ValueError
            If API key is not provided and not found in environment.
        Exception
            If connection to Pinecone fails.
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key is required")

        self.host = host
        self.index_name = index_name

        try:
            # Initialize Pinecone client
            pc = Pinecone(api_key=self.api_key)
            self.index = pc.Index(index_name=self.index_name, host=self.host)

            # Test connection
            self.index.describe_index_stats()

            logger.info(
                f"Successfully connected to Pinecone index '{self.index_name}' "
                f"with host '{self.host}'"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise

    def upsert_embeddings(
        self, embeddings: List[Dict[str, Any]], namespace: str = ""
    ) -> None:
        """
        Upsert embeddings into the vector database.

        Parameters
        ----------
        embeddings : List[Dict[str, Any]]
            List of embeddings to upsert. Each dict should contain:
            - 'id': Unique identifier for the vector
            - 'embedding': Vector embedding as list of floats
            - 'metadata': Optional metadata dictionary
        namespace : str, default=""
            Namespace for the embeddings.

        Raises
        ------
        ValueError
            If embeddings list is empty or malformed.
        Exception
            If upsert operation fails.
        """
        if not embeddings:
            raise ValueError("Embeddings list cannot be empty")

        try:
            # Prepare vectors for upsert
            vectors = []
            for item in embeddings:
                if "id" not in item or "embedding" not in item:
                    raise ValueError(
                        "Each embedding must have 'id' and 'embedding' keys"
                    )

                vector_id = item["id"]
                embedding = item["embedding"]
                metadata = item.get("metadata", {})

                # Create vector tuple
                vector = (vector_id, embedding, metadata)
                vectors.append(vector)

            # Upsert vectors
            self.index.upsert(vectors=vectors, namespace=namespace)

            logger.info(
                f"Successfully upserted {len(vectors)} embeddings into index "
                f"'{self.index_name}' with namespace '{namespace}'"
            )

        except Exception as e:
            logger.error(f"Failed to upsert embeddings: {e}")
            raise

    def query(
        self,
        embedding: List[float],
        top_k: int = 1,
        filters: Optional[Dict[str, Any]] = None,
        namespace: str = "",
        include_values: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the vector database to find similar items.

        Parameters
        ----------
        embedding : List[float]
            The query embedding vector.
        top_k : int, default=1
            Number of top results to retrieve.
        filters : Dict[str, Any], optional
            Metadata filters to apply to the query.
        namespace : str, default=""
            Namespace to query within.
        include_values : bool, default=False
            Whether to include embedding values in the results.

        Returns
        -------
        Dict[str, Any]
            Query results from Pinecone containing:
            - 'matches': List of similar vectors with scores
            - 'namespace': Namespace that was queried
            - 'usage': Query usage statistics

        Raises
        ------
        ValueError
            If embedding is empty or invalid.
        Exception
            If query operation fails.
        """
        if not embedding:
            raise ValueError("Query embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            query_result = self.index.query(
                vector=embedding,
                top_k=top_k,
                filter=filters,
                namespace=namespace,
                include_values=include_values,
                include_metadata=True,
            )

            logger.debug(
                f"Queried index '{self.index_name}' with namespace '{namespace}', "
                f"found {len(query_result.get('matches', []))} matches"
            )

            return query_result

        except Exception as e:
            logger.error(f"Failed to query vector database: {e}")
            raise

    def delete_vectors(self, ids: List[str], namespace: str = "") -> None:
        """
        Delete vectors from the database.

        Parameters
        ----------
        ids : List[str]
            List of vector IDs to delete.
        namespace : str, default=""
            Namespace containing the vectors.
        """
        if not ids:
            logger.warning("No vector IDs provided for deletion")
            return

        try:
            self.index.delete(ids=ids, namespace=namespace)
            logger.info(
                f"Deleted {len(ids)} vectors from index '{self.index_name}' "
                f"with namespace '{namespace}'"
            )
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.

        Returns
        -------
        Dict[str, Any]
            Index statistics including:
            - 'total_vector_count': Total number of vectors
            - 'namespaces': Statistics per namespace
            - 'dimension': Vector dimension
        """
        try:
            stats = self.index.describe_index_stats()
            logger.debug(f"Retrieved stats for index '{self.index_name}': {stats}")
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            raise

    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in the index.

        Returns
        -------
        List[str]
            List of namespace names.
        """
        try:
            stats = self.get_index_stats()
            namespaces = list(stats.get("namespaces", {}).keys())
            logger.debug(f"Found namespaces in index '{self.index_name}': {namespaces}")
            return namespaces
        except Exception as e:
            logger.error(f"Failed to list namespaces: {e}")
            raise

    def clear_namespace(self, namespace: str) -> None:
        """
        Clear all vectors from a specific namespace.

        Parameters
        ----------
        namespace : str
            Namespace to clear.
        """
        try:
            self.index.delete(namespace=namespace, delete_all=True)
            logger.info(
                f"Cleared all vectors from namespace '{namespace}' "
                f"in index '{self.index_name}'"
            )
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Pinecone doesn't require explicit cleanup
        pass


if __name__ == "__main__":
    vector_db_style = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_STYLE"),
        index_name="catalog-style-description",
    )
