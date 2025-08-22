"""
Neo4j graph database handler for fashion recommendation system.

This module provides the GraphDatabaseHandler class for managing
fashion product relationships in a Neo4j graph database.
"""

from typing import Dict, List, Any, Optional
from neo4j import GraphDatabase
from loguru import logger


class GraphDatabaseHandler:
    """
    Handler for Neo4j graph database operations.

    This class manages all interactions with the Neo4j graph database,
    including creating product nodes, managing relationships, and
    retrieving recommendations.
    """

    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize the graph database handler.

        Parameters
        ----------
        uri : str
            Neo4j database URI (e.g., "bolt://localhost:7687").
        user : str
            Neo4j username.
        password : str
            Neo4j password.

        Raises
        ------
        Exception
            If connection to Neo4j fails.
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Successfully connected to Neo4j database at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j database: {e}")
            raise

    def close(self) -> None:
        """Close the database connection."""
        if hasattr(self, "driver"):
            self.driver.close()
            logger.info("Closed connection to Neo4j database")

    def create_product_node(self, product_id: str, attributes: Dict[str, Any]) -> None:
        """
        Create or update a product node in the graph.

        Parameters
        ----------
        product_id : str
            Unique identifier for the product.
        attributes : Dict[str, Any]
            Product attributes to store as node properties.
        """
        with self.driver.session() as session:
            session.write_transaction(
                self._create_and_return_node, product_id, attributes
            )
            logger.info(f"Created/Updated product node for product_id: {product_id}")

    @staticmethod
    def _create_and_return_node(
        tx, product_id: str, attributes: Dict[str, Any]
    ) -> None:
        """
        Create or update a product node in a transaction.

        Parameters
        ----------
        tx : neo4j.Transaction
            Neo4j transaction object.
        product_id : str
            Unique identifier for the product.
        attributes : Dict[str, Any]
            Product attributes to store as node properties.
        """
        query = """
            MERGE (p:Product {product_id: $product_id})
            SET p += $attributes
        """
        logger.debug(
            f"Executing query: {query} with parameters: product_id={product_id}"
        )
        tx.run(query, product_id=product_id, attributes=attributes)

    def create_or_update_relationship(
        self,
        product_id1: str,
        product_id2: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create or update a relationship between two products.

        Parameters
        ----------
        product_id1 : str
            ID of the first product.
        product_id2 : str
            ID of the second product.
        relationship_type : str
            Type of relationship (e.g., "WORN_WITH", "COMPLEMENTED_BY").
        properties : Dict[str, Any], optional
            Properties to store on the relationship.
        """
        if properties is None:
            properties = {}

        with self.driver.session() as session:
            session.write_transaction(
                self._create_or_update_relationship,
                product_id1,
                product_id2,
                relationship_type,
                properties,
            )
            logger.info(
                f"Created/Updated {relationship_type} relationship between "
                f"{product_id1} and {product_id2}"
            )

    @staticmethod
    def _create_or_update_relationship(
        tx,
        product_id1: str,
        product_id2: str,
        relationship_type: str,
        properties: Dict[str, Any],
    ) -> None:
        """
        Create or update a relationship in a transaction.

        This method handles relationship creation with special handling for:
        - weight: incremented on MATCH, set to 1 on CREATE
        - image: stored as arrays, extended on MATCH
        - other properties: overwritten on both CREATE and MATCH

        Parameters
        ----------
        tx : neo4j.Transaction
            Neo4j transaction object.
        product_id1 : str
            ID of the first product.
        product_id2 : str
            ID of the second product.
        relationship_type : str
            Type of relationship.
        properties : Dict[str, Any]
            Properties to store on the relationship.
        """
        # Prepare separate assignment strings for ON CREATE and ON MATCH
        create_assignments = []
        match_assignments = []

        # Handle properties except weight (handled separately)
        for key, _ in properties.items():
            if key == "weight":
                continue  # Skip weight (handled separately)
            elif key == "image":
                # Store as an array, extend on match
                create_assignments.append(f"r.image = [${key}]")
                match_assignments.append(f"r.image = r.image + [${key}]")
            else:
                # Overwrite on create and match
                create_assignments.append(f"r.{key} = ${key}")
                match_assignments.append(f"r.{key} = ${key}")

        # Build the final string fragments
        properties_str_create = (
            (", " + ", ".join(create_assignments)) if create_assignments else ""
        )
        properties_str_match = (
            (", " + ", ".join(match_assignments)) if match_assignments else ""
        )

        # Build the MERGE query
        query = f"""
            MATCH (p1:Product {{product_id: $product_id1}})
            MATCH (p2:Product {{product_id: $product_id2}})
            MERGE (p1)-[r:{relationship_type}]->(p2)
            ON CREATE SET r.weight = 1
            {properties_str_create}
            ON MATCH SET r.weight = r.weight + 1
            {properties_str_match}
        """

        # Combine parameters
        params = {"product_id1": product_id1, "product_id2": product_id2, **properties}

        logger.debug(f"Executing query: {query} with parameters: {params}")
        tx.run(query, **params)

    def get_recommendations(
        self,
        selected_product_id: str,
        filters: Dict[str, Any],
        threshold: int,
        top_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get product recommendations based on graph relationships.

        Parameters
        ----------
        selected_product_id : str
            ID of the product to get recommendations for.
        filters : Dict[str, Any]
            Filters to apply to recommendations (e.g., type, gender).
        threshold : int
            Minimum weight threshold for relationships to consider.
        top_k : int
            Maximum number of recommendations to return per category.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary containing:
            - 'selected_results': List with selected product info
            - 'worn_with': List of products worn with the selected product
            - 'complemented': List of products that complement the selected product
        """
        with self.driver.session() as session:
            result = session.read_transaction(
                self._get_recommendations_tx,
                selected_product_id,
                filters,
                threshold,
                top_k,
            )
            logger.info(
                f"Retrieved recommendations for product_id: {selected_product_id}"
            )
            return result

    @staticmethod
    def _get_recommendations_tx(
        tx,
        selected_product_id: str,
        filters: Dict[str, Any],
        threshold: int,
        top_k: int,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get recommendations in a transaction.

        Parameters
        ----------
        tx : neo4j.Transaction
            Neo4j transaction object.
        selected_product_id : str
            ID of the product to get recommendations for.
        filters : Dict[str, Any]
            Filters to apply to recommendations.
        threshold : int
            Minimum weight threshold for relationships.
        top_k : int
            Maximum number of recommendations to return.

        Returns
        -------
        Dict[str, List[Dict[str, Any]]]
            Dictionary with recommendation results.
        """
        # Fetch the type of the selected product
        query_get_type = """
            MATCH (p:Product {product_id: $selected_product_id})
            RETURN p.type AS selected_type, properties(p) AS metadata
        """
        logger.debug(
            f"Executing query: {query_get_type} with parameters: selected_product_id={selected_product_id}"
        )

        result = tx.run(query_get_type, selected_product_id=selected_product_id)
        record = result.single()

        if record is None:
            logger.error(f"No product found with product_id: {selected_product_id}")
            return {"selected_results": [], "worn_with": [], "complemented": []}

        selected_type = record["selected_type"]

        # Build filter conditions
        filter_conditions = " AND ".join([f"related.{key} = ${key}" for key in filters])

        params = {
            "selected_product_id": selected_product_id,
            "selected_type": selected_type,
            "threshold": threshold,
            "top_k": top_k,
            **filters,
        }

        # Query for 'WORN_WITH' relationships (different types)
        query_worn_with = f"""
            MATCH (p:Product {{product_id: $selected_product_id}})-[r:WORN_WITH]-(related:Product)
            WHERE r.weight >= $threshold AND related.type <> $selected_type
            {"AND " + filter_conditions if filter_conditions else ""}
            RETURN related.product_id AS product_id, r.weight AS weight, 
                   r.image AS images, r.source AS source, properties(related) AS metadata
            ORDER BY r.weight DESC
            LIMIT $top_k
        """

        # Query for 'COMPLEMENTED_BY' relationships (same type)
        query_complemented = f"""
            MATCH (p:Product {{product_id: $selected_product_id}})-[r:COMPLEMENTED_BY]-(related:Product)
            WHERE r.weight >= $threshold AND related.type = $selected_type
            {"AND " + filter_conditions if filter_conditions else ""}
            RETURN related.product_id AS product_id, r.weight AS weight, 
                   r.image AS images, r.source AS source, properties(related) AS metadata
            ORDER BY r.weight DESC
            LIMIT $top_k
        """

        logger.debug(
            f"Executing worn_with query: {query_worn_with} with parameters: {params}"
        )
        logger.debug(
            f"Executing complemented query: {query_complemented} with parameters: {params}"
        )

        # Execute queries
        worn_with_result = tx.run(query_worn_with, **params)
        complemented_result = tx.run(query_complemented, **params)

        # Convert results to lists of dictionaries
        selected_results = [dict(record)]
        worn_with_recommendations = [dict(record) for record in worn_with_result]
        complemented_recommendations = [dict(record) for record in complemented_result]

        logger.debug(
            f"Found {len(worn_with_recommendations)} worn_with recommendations"
        )
        logger.debug(
            f"Found {len(complemented_recommendations)} complemented recommendations"
        )

        return {
            "selected_results": selected_results,
            "worn_with": worn_with_recommendations,
            "complemented": complemented_recommendations,
        }

    def get_product_info(self, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific product.

        Parameters
        ----------
        product_id : str
            ID of the product to retrieve.

        Returns
        -------
        Optional[Dict[str, Any]]
            Product information or None if not found.
        """
        with self.driver.session() as session:
            result = session.read_transaction(self._get_product_info_tx, product_id)
            return result

    @staticmethod
    def _get_product_info_tx(tx, product_id: str) -> Optional[Dict[str, Any]]:
        """
        Get product information in a transaction.

        Parameters
        ----------
        tx : neo4j.Transaction
            Neo4j transaction object.
        product_id : str
            ID of the product to retrieve.

        Returns
        -------
        Optional[Dict[str, Any]]
            Product information or None if not found.
        """
        query = """
            MATCH (p:Product {product_id: $product_id})
            RETURN properties(p) AS properties
        """
        result = tx.run(query, product_id=product_id)
        record = result.single()
        return dict(record["properties"]) if record else None

    def delete_product(self, product_id: str) -> bool:
        """
        Delete a product and all its relationships.

        Parameters
        ----------
        product_id : str
            ID of the product to delete.

        Returns
        -------
        bool
            True if product was deleted, False if not found.
        """
        with self.driver.session() as session:
            result = session.write_transaction(self._delete_product_tx, product_id)
            if result:
                logger.info(f"Deleted product: {product_id}")
            else:
                logger.warning(f"Product not found for deletion: {product_id}")
            return result

    @staticmethod
    def _delete_product_tx(tx, product_id: str) -> bool:
        """
        Delete a product in a transaction.

        Parameters
        ----------
        tx : neo4j.Transaction
            Neo4j transaction object.
        product_id : str
            ID of the product to delete.

        Returns
        -------
        bool
            True if product was deleted, False if not found.
        """
        query = """
            MATCH (p:Product {product_id: $product_id})
            DETACH DELETE p
            RETURN count(p) as deleted_count
        """
        result = tx.run(query, product_id=product_id)
        record = result.single()
        return record["deleted_count"] > 0 if record else False

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
