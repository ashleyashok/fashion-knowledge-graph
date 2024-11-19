# src/database/vector_database.py

from pinecone import Pinecone
from typing import List, Dict
import os


class VectorDatabase:
    """
    A class to handle interactions with the vector database (Pinecone).
    """

    def __init__(
        self, api_key: str = None, host: str = None, index_name: str = "catalog-clothes"
    ):
        self.pc = Pinecone()
        self.index = self.pc.Index(index_name)

    def upsert_embeddings(self, embeddings: List[Dict], namespace: str = ""):
        """
        Upsert embeddings into the vector database.
        Each embedding dict should contain 'id', 'embedding', and 'metadata'.
        Optionally specify a namespace.
        """
        vectors = []
        for item in embeddings:
            vector_id = item["id"]
            vector = {
                "id": vector_id,
                "values": item["embedding"],
                "metadata": item["metadata"],
            }
            vectors.append(vector)
        self.index.upsert(vectors=vectors, namespace=namespace)

    def query(
        self,
        embedding: List[float],
        top_k: int = 1,
        filters: Dict = None,
        namespace: str = "",
        include_values: bool = False,
    ):
        """
        Query the vector database to find similar items.
        """
        query_result = self.index.query(
            vector=embedding,
            top_k=top_k,
            filter=filters,
            namespace=namespace,
            include_values=include_values,
            include_metadata=True,
        )
        return query_result


if __name__ == "__main__":
    vdb = VectorDatabase(
        index_name="catalog-clothes",
    )
