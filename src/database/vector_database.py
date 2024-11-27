# src/database/vector_database.py

from pinecone import Pinecone
from typing import List, Dict, Optional
import os
from loguru import logger


class VectorDatabase:
    """
    A class to handle interactions with the vector database (Pinecone).
    """

    def __init__(
        self,
        host: str,
        index_name: str = "catalog-clothes",
        api_key: Optional[str] = None,
    ):
        """
        Initializes the vector database with the given parameters.

        Args:
            host (str): The host address for the Pinecone service.
            index_name (str, optional): The name of the index to use. Defaults to "catalog-clothes".
            api_key (Optional[str], optional): The API key for authentication. If not provided, it will be fetched from the environment variable 'PINECONE_API_KEY'.

        Attributes:
            api_key (str): The API key for Pinecone service.
            host (str): The host address for the Pinecone service.
            index_name (str): The name of the index to use.
            index (Pinecone.Index): The Pinecone index object initialized with the given parameters.
        """

        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.host = host
        self.index_name = index_name

        pc = Pinecone(api_key=self.api_key)
        self.index = pc.Index(index_name=self.index_name, host=self.host)
        logger.info(
            f"Initialized Pinecone Index '{self.index_name}' with host '{self.host}'"
        )

    def upsert_embeddings(self, embeddings: List[Dict], namespace: str = ""):
        """
        Upsert embeddings into the vector database.
        Each embedding dict should contain 'id', 'embedding', and 'metadata'.
        Optionally specify a namespace.

        Parameters
        ----------
        embeddings : List[Dict]
            List of embeddings to upsert.
        namespace : str, optional
            Namespace for the embeddings. Default is empty string.
        """
        vectors = []
        for item in embeddings:
            vector_id = item["id"]
            vector = (vector_id, item["embedding"], item["metadata"])
            vectors.append(vector)
        self.index.upsert(vectors=vectors, namespace=namespace)
        logger.info(
            f"Upserted {len(vectors)} embeddings into index '{self.index_name}' with namespace '{namespace}'"
        )

    def query(
        self,
        embedding: List[float],
        top_k: int = 1,
        filters: Optional[Dict] = None,
        namespace: str = "",
        include_values: bool = False,
    ):
        """
        Query the vector database to find similar items.

        Parameters
        ----------
        embedding : List[float]
            The query embedding vector.
        top_k : int, optional
            Number of top results to retrieve. Default is 1.
        filters : Dict, optional
            Metadata filters to apply to the query.
        namespace : str, optional
            Namespace to query within.
        include_values : bool, optional
            Whether to include embedding values in the results.

        Returns
        -------
        Dict
            Query results from Pinecone.
        """
        query_result = self.index.query(
            vector=embedding,
            top_k=top_k,
            filter=filters,
            namespace=namespace,
            include_values=include_values,
            include_metadata=True,
        )
        logger.info(f"Queried index '{self.index_name}' with namespace '{namespace}'")
        return query_result


if __name__ == "__main__":
    vector_db_style = VectorDatabase(
        api_key=os.getenv("PINECONE_API_KEY"),
        host=os.getenv("PINECONE_HOST_STYLE"),
        index_name="catalog-style-description",
    )
