"""ArangoDB schema inspection for the Arango backend."""

from dataclasses import asdict, dataclass
import json
from typing import Any, Dict, List, Optional


@dataclass
class CollectionSchema:
    """Schema information for a single document collection."""

    name: str
    type: str
    count: int
    fields: List[str]


@dataclass
class EdgeSchema:
    """Schema information for a single edge collection."""

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


class ArangoSchemaInspector:
    """Discover ArangoDB collection and edge shapes from sampled documents."""

    def __init__(self, db, max_sample_docs: int = 5):
        self.db = db
        self.max_sample_docs = max_sample_docs
        self._schema_cache: Optional[SchemaMap] = None

    def get_schema_map(self, refresh: bool = False) -> Dict[str, Any]:
        if self._schema_cache is None or refresh:
            self._schema_cache = self._build_schema_map()
        return asdict(self._schema_cache)

    def get_collection_info(
        self, collection_name: str
    ) -> Optional[Dict[str, Any]]:
        schema = self.get_schema_map()
        for collection in schema["collections"]:
            if collection["name"] == collection_name:
                return collection
        for edge in schema["edges"]:
            if edge["name"] == collection_name:
                return edge
        return None

    def get_edge_info(self, edge_collection: str) -> Optional[Dict[str, Any]]:
        for edge in self.get_schema_map()["edges"]:
            if edge["name"] == edge_collection:
                return edge
        return None

    def _build_schema_map(self) -> SchemaMap:
        document_collections = []
        edge_collections = []
        for collection_info in self.db.collections():
            collection_name = collection_info["name"]
            if collection_name.startswith("_"):
                continue
            if collection_info["type"] == 3:
                edge_collections.append(
                    self._inspect_edge_collection(collection_name)
                )
            else:
                document_collections.append(
                    self._inspect_document_collection(collection_name)
                )
        return SchemaMap(
            database_name=self.db.name,
            collections=document_collections,
            edges=edge_collections,
        )

    def _inspect_document_collection(
        self, collection_name: str
    ) -> CollectionSchema:
        collection = self.db.collection(collection_name)
        count = collection.count()
        fields = set()
        if count > 0:
            cursor = self.db.aql.execute(
                f"""
                FOR doc IN {collection_name}
                LIMIT {max(1, self.max_sample_docs)}
                RETURN doc
                """
            )
            for document in cursor:
                fields.update(document.keys())
        return CollectionSchema(
            name=collection_name,
            type="document",
            count=count,
            fields=sorted(fields),
        )

    def _inspect_edge_collection(self, collection_name: str) -> EdgeSchema:
        collection = self.db.collection(collection_name)
        count = collection.count()
        fields = set()
        from_collections = set()
        to_collections = set()
        if count > 0:
            cursor = self.db.aql.execute(
                f"""
                FOR edge IN {collection_name}
                LIMIT {max(1, self.max_sample_docs)}
                RETURN edge
                """
            )
            for edge in cursor:
                fields.update(edge.keys())
                if "_from" in edge:
                    from_collections.add(edge["_from"].split("/")[0])
                if "_to" in edge:
                    to_collections.add(edge["_to"].split("/")[0])
        return EdgeSchema(
            name=collection_name,
            count=count,
            from_collections=sorted(from_collections),
            to_collections=sorted(to_collections),
            fields=sorted(fields),
        )


def inspect_arango_schema(
    db, output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Inspect an ArangoDB schema and optionally write its map to JSON."""
    schema = ArangoSchemaInspector(db).get_schema_map()
    if output_file:
        with open(output_file, "w", encoding="utf-8") as stream:
            json.dump(schema, stream, indent=2, default=str)
    return schema
