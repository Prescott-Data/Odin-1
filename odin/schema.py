"""
Used by AI agents to understand graph structure and write valid AQL queries.
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class CollectionSchema:
    """Schema information for a single collection."""
    name: str
    type: str  # "document" or "edge"
    count: int
    fields: List[str]


@dataclass
class EdgeSchema:
    """Schema information for an edge collection."""
    name: str
    count: int
    from_collections: List[str]
    to_collections: List[str]
    fields: List[str]


@dataclass
class SchemaMap:
    """Complete schema map of an ArangoDB database."""
    database_name: str
    collections: List[CollectionSchema]
    edges: List[EdgeSchema]  


class SchemaInspector:
    """
    
    Queries the database to discover:
    - All collections (vertex and edge)
    - Field names in each collection
    - Edge relationships (_from/_to patterns)
    
    Usage:
        inspector = SchemaInspector(arango_db)
        schema = inspector.get_schema_map()
        entity_info = inspector.get_collection_info("ExtractedEntities")
    """
    
    def __init__(self, db, max_sample_docs: int = 5):
        """
        Initialize schema inspector.
        """
        self.db = db
        self.max_sample_docs = max_sample_docs
        self._schema_cache: Optional[SchemaMap] = None
    
    def get_schema_map(self, refresh: bool = False) -> Dict[str, Any]:
        if self._schema_cache is None or refresh:
            self._schema_cache = self._build_schema_map()
        
        return asdict(self._schema_cache)
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        schema = self.get_schema_map()
        
        # Check document collections
        for col in schema['collections']:
            if col['name'] == collection_name:
                return col
        
        # Check edge collections
        for edge in schema['edges']:
            if edge['name'] == collection_name:
                return edge
        
        return None
    
    def get_edge_info(self, edge_collection: str) -> Optional[Dict[str, Any]]:
        schema = self.get_schema_map()
        
        for edge in schema['edges']:
            if edge['name'] == edge_collection:
                return edge
        
        return None
    
    def _build_schema_map(self) -> SchemaMap:
        """Build complete schema map by querying ArangoDB."""
        db_name = self.db.name
        
        # Get all collections
        all_collections = self.db.collections()
        
        document_collections = []
        edge_collections = []
        
        for col_info in all_collections:
            col_name = col_info['name']
            
            # Skip system collections
            if col_name.startswith('_'):
                continue
            
            col = self.db.collection(col_name)
            is_edge = col_info['type'] == 3  # Edge collection type
            
            if is_edge:
                edge_schema = self._inspect_edge_collection(col_name)
                edge_collections.append(edge_schema)
            else:
                doc_schema = self._inspect_document_collection(col_name)
                document_collections.append(doc_schema)
        
        return SchemaMap(
            database_name=db_name,
            collections=document_collections,
            edges=edge_collections
        )
    
    def _inspect_document_collection(self, col_name: str) -> CollectionSchema:
        """Inspect a document collection and extract schema."""
        col = self.db.collection(col_name)
        count = col.count()
        
        # Get sample documents to extract fields (always fetch at least 1 for field discovery)
        fields = set()
        
        if count > 0:
            # Use max(1, max_sample_docs) to ensure at least 1 doc for fields
            sample_limit = max(1, self.max_sample_docs)
            aql = f"""
            FOR doc IN {col_name}
            LIMIT {sample_limit}
            RETURN doc
            """
            cursor = self.db.aql.execute(aql)
            
            for doc in cursor:
                # Extract all field names
                fields.update(doc.keys())
        
        return CollectionSchema(
            name=col_name,
            type="document",
            count=count,
            fields=sorted(list(fields))
        )
    
    def _inspect_edge_collection(self, col_name: str) -> EdgeSchema:
        """Inspect an edge collection and extract schema."""
        col = self.db.collection(col_name)
        count = col.count()
        
        # Get sample edges to extract fields and _from/_to patterns (always fetch at least 1)
        fields = set()
        from_collections = set()
        to_collections = set()
        
        if count > 0:
            # Use max(1, max_sample_docs) to ensure at least 1 edge for fields
            sample_limit = max(1, self.max_sample_docs)
            aql = f"""
            FOR edge IN {col_name}
            LIMIT {sample_limit}
            RETURN edge
            """
            cursor = self.db.aql.execute(aql)
            
            for edge in cursor:
                # Extract fields
                fields.update(edge.keys())
                
                # Extract _from/_to collection names
                if '_from' in edge:
                    from_col = edge['_from'].split('/')[0]
                    from_collections.add(from_col)
                
                if '_to' in edge:
                    to_col = edge['_to'].split('/')[0]
                    to_collections.add(to_col)
        
        return EdgeSchema(
            name=col_name,
            count=count,
            from_collections=sorted(list(from_collections)),
            to_collections=sorted(list(to_collections)),
            fields=sorted(list(fields))
        )


def inspect_arango_schema(db, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to inspect ArangoDB schema and optionally save to file.
    
    Args:
        db: ArangoDB database connection
        output_file: Optional path to save schema as JSON
        
    Returns:
        Schema map as dictionary
    """
    inspector = SchemaInspector(db)
    schema = inspector.get_schema_map()
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(schema, f, indent=2, default=str)
    
    return schema
