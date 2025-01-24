# src/database/graph_database.py

from neo4j import GraphDatabase
from typing import Dict, List, Any
from loguru import logger

class GraphDatabaseHandler:
    """
    A class to handle interactions with Neo4j graph database.
    """

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j database at {uri}")

    def close(self):
        self.driver.close()
        logger.info("Closed connection to Neo4j database")

    def create_product_node(self, product_id: str, attributes: Dict[str, Any]):
        with self.driver.session() as session:
            session.write_transaction(self._create_and_return_node, product_id, attributes)
            logger.info(f"Created/Updated product node for product_id: {product_id}")

    @staticmethod
    def _create_and_return_node(tx, product_id: str, attributes: Dict[str, Any]):
        query = """
            MERGE (p:Product {product_id: $product_id})
            SET p += $attributes
        """
        logger.info(f"Executing query: {query} with parameters: product_id={product_id}, attributes={attributes}")
        tx.run(query, product_id=product_id, attributes=attributes)

    def create_or_update_relationship(
        self,
        product_id1: str,
        product_id2: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ):
        with self.driver.session() as session:
            session.write_transaction(
                self._create_or_update_relationship,
                product_id1,
                product_id2,
                relationship_type,
                properties,
            )
            logger.info(f"Created/Updated {relationship_type} relationship between {product_id1} and {product_id2}")

    @staticmethod
    def _create_or_update_relationship(
        tx,
        product_id1: str,
        product_id2: str,
        relationship_type: str,
        properties: Dict[str, Any] = None,
    ):
        """
        Create or update a relationship between two Product nodes.
        
        - weight is incremented on MATCH, set to 1 on CREATE
        - image is stored/updated as arrays:
            * On CREATE: r.image = [$image]
            * On MATCH:  r.image = r.image + [$image]
        - All other properties (e.g. 'source') are overwritten on CREATE and MATCH.
        
        Args:
            tx: Neo4j transaction object.
            product_id1 (str): The ID of the first product.
            product_id2 (str): The ID of the second product.
            relationship_type (str): The type of the relationship (e.g., "WORN_WITH").
            properties (Dict[str, Any], optional): Additional properties to set on the relationship.
        """

        # Ensure properties is at least an empty dict
        if not properties:
            properties = {}

        # Prepare separate assignment strings for ON CREATE and ON MATCH
        create_assignments = []
        match_assignments = []

        # We'll handle 'weight' specially in our MERGE query,
        # so do not add it to the property assignments here.
        for key, _ in properties.items():
            if key == "weight":
                # Skip weight (handled separately below)
                continue
            elif key == "image":
                # Store as an array, extend on match
                create_assignments.append(f"r.image = [${key}]")
                match_assignments.append(f"r.image = r.image + [${key}]")
            else:
                # Overwrite on create and match
                create_assignments.append(f"r.{key} = ${key}")
                match_assignments.append(f"r.{key} = ${key}")

        # Build the final string fragments for our Cypher query
        properties_str_create = (
            (", " + ", ".join(create_assignments)) if create_assignments else ""
        )
        properties_str_match = (
            (", " + ", ".join(match_assignments)) if match_assignments else ""
        )

        # Now build our MERGE query
        # - weight is set to 1 on CREATE
        # - weight is incremented on MATCH
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
        params = {'product_id1': product_id1, 'product_id2': product_id2, **properties}

        logger.info(f"Executing query: {query} with parameters: {params}")
        tx.run(query, **params)

    def get_recommendations(self, selected_product_id: str, filters: Dict[str, Any], threshold: int, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        with self.driver.session() as session:
            result = session.read_transaction(
                self._get_recommendations_tx,
                selected_product_id,
                filters,
                threshold,
                top_k
            )
            logger.info(f"Retrieved recommendations for product_id: {selected_product_id}")
            return result

    @staticmethod
    def _get_recommendations_tx(tx, selected_product_id: str, filters: Dict[str, Any], threshold: int, top_k: int):
        # Fetch the type of the selected product
        query_get_type = """
            MATCH (p:Product {product_id: $selected_product_id})
            RETURN p.type AS selected_type, properties(p) AS metadata
        """
        logger.info(f"Executing query: {query_get_type} with parameters: selected_product_id={selected_product_id}")
        result = tx.run(query_get_type, selected_product_id=selected_product_id)
        record = result.single()
        if record is None:
            logger.error(f"No product found with product_id: {selected_product_id}")
            return {'selected_results': [],'worn_with': [], 'complemented': []}
        selected_type = record["selected_type"]

        # Build the filter conditions
        filter_conditions = ' AND '.join([f'related.{key} = ${key}' for key in filters])

        params = {
            'selected_product_id': selected_product_id,
            'selected_type': selected_type,
            'threshold': threshold,
            'top_k': top_k
        }
        params.update(filters)

        # Query for 'WORN_WITH' relationships (different types)
        query_worn_with = f"""
            MATCH (p:Product {{product_id: $selected_product_id}})-[r:WORN_WITH]-(related:Product)
            WHERE r.weight >= $threshold AND related.type <> $selected_type
            {"AND " + filter_conditions if filter_conditions else ""}
            RETURN related.product_id AS product_id, r.weight AS weight, r.image AS images, r.source AS source, properties(related) AS metadata
            ORDER BY r.weight DESC
            LIMIT $top_k
        """
        logger.info(f"Executing query: {query_worn_with} with parameters: {params}")

        # Query for 'COMPLEMENTED_BY' relationships (same type)
        query_complemented = f"""
            MATCH (p:Product {{product_id: $selected_product_id}})-[r:COMPLEMENTED_BY]-(related:Product)
            WHERE r.weight >= $threshold AND related.type = $selected_type
            {"AND " + filter_conditions if filter_conditions else ""}
            RETURN related.product_id AS product_id, r.weight AS weight, r.image AS images, r.source AS source, properties(related) AS metadata
            ORDER BY r.weight DESC
            LIMIT $top_k
        """
        logger.info(f"Executing query: {query_complemented} with parameters: {params}")

        # Execute queries
        worn_with_result = tx.run(query_worn_with, **params)
        complemented_result = tx.run(query_complemented, **params)

        # Convert results to lists of dictionaries
        selected_results = [dict(record)]
        worn_with_recommendations = [dict(record) for record in worn_with_result]
        complemented_recommendations = [dict(record) for record in complemented_result]

        logger.info(f"Worn with recommendations: {worn_with_recommendations}")
        logger.info(f"Complemented recommendations: {complemented_recommendations}")

        return {
            'selected_results': selected_results,
            'worn_with': worn_with_recommendations,
            'complemented': complemented_recommendations
        }
