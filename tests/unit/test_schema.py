"""
Unit tests for ArangoDB Schema Inspector
"""
import pytest
from unittest.mock import Mock, MagicMock
from odin.schema import SchemaInspector, CollectionSchema, EdgeSchema, inspect_arango_schema


@pytest.fixture
def mock_db():
    """Create a mock ArangoDB database."""
    db = Mock()
    db.name = "test_db"
    return db


@pytest.fixture
def mock_collections():
    """Mock collection metadata."""
    return [
        {'name': 'ExtractedEntities', 'type': 2},  # Document collection
        {'name': 'ExtractedRelationships', 'type': 3},  # Edge collection
        {'name': 'Documents', 'type': 2},
        {'name': 'EXTRACTED_FROM', 'type': 3},
        {'name': '_system', 'type': 2},  # System collection (should be ignored)
    ]


@pytest.fixture
def mock_entity_docs():
    """Mock entity documents."""
    return [
        {
            '_key': 'entity1',
            '_id': 'ExtractedEntities/entity1',
            'name': 'John Doe',
            'type': 'Person',
            'created_at': '2024-01-01T00:00:00Z'
        },
        {
            '_key': 'entity2',
            '_id': 'ExtractedEntities/entity2',
            'name': 'ACME Corp',
            'type': 'Organization',
            'created_at': '2024-01-02T00:00:00Z'
        }
    ]


@pytest.fixture
def mock_relationship_edges():
    """Mock relationship edges."""
    return [
        {
            '_key': 'edge1',
            '_id': 'ExtractedRelationships/edge1',
            '_from': 'ExtractedEntities/entity1',
            '_to': 'ExtractedEntities/entity2',
            'relationship': 'works_at',
            'created_at': '2024-01-01T00:00:00Z',
            'raw_confidence': 0.95
        },
        {
            '_key': 'edge2',
            '_id': 'ExtractedRelationships/edge2',
            '_from': 'ExtractedEntities/entity2',
            '_to': 'Documents/doc1',
            'relationship': 'mentioned_in',
            'created_at': '2024-01-02T00:00:00Z',
            'raw_confidence': 0.88
        }
    ]


class TestSchemaInspector:
    """Test cases for SchemaInspector class."""
    
    def test_init(self, mock_db):
        """Test SchemaInspector initialization."""
        inspector = SchemaInspector(mock_db, max_sample_docs=10)
        assert inspector.db == mock_db
        assert inspector.max_sample_docs == 10
        assert inspector._schema_cache is None
    
    def test_get_schema_map_caching(self, mock_db, mock_collections, mock_entity_docs):
        """Test that schema map is cached."""
        # Setup mocks
        mock_db.collections.return_value = mock_collections
        
        # Mock collection objects
        entity_col = Mock()
        entity_col.count.return_value = 2
        
        # Mock AQL cursor
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_entity_docs))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = entity_col
        
        inspector = SchemaInspector(mock_db)
        
        # First call should query database
        schema1 = inspector.get_schema_map()
        assert schema1 is not None
        assert mock_db.collections.call_count == 1
        
        # Second call should use cache
        schema2 = inspector.get_schema_map()
        assert schema2 == schema1
        assert mock_db.collections.call_count == 1  # Not called again
        
        # Refresh should query again
        schema3 = inspector.get_schema_map(refresh=True)
        assert mock_db.collections.call_count == 2
    
    def test_inspect_document_collection(self, mock_db, mock_entity_docs):
        """Test document collection inspection."""
        # Setup mocks
        entity_col = Mock()
        entity_col.count.return_value = len(mock_entity_docs)
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_entity_docs))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = entity_col
        
        inspector = SchemaInspector(mock_db)
        col_schema = inspector._inspect_document_collection('ExtractedEntities')
        
        assert col_schema.name == 'ExtractedEntities'
        assert col_schema.type == 'document'
        assert col_schema.count == 2
        assert '_key' in col_schema.fields
        assert 'name' in col_schema.fields
        assert 'type' in col_schema.fields
    
    def test_inspect_edge_collection(self, mock_db, mock_relationship_edges):
        """Test edge collection inspection."""
        # Setup mocks
        edge_col = Mock()
        edge_col.count.return_value = len(mock_relationship_edges)
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_relationship_edges))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = edge_col
        
        inspector = SchemaInspector(mock_db)
        edge_schema = inspector._inspect_edge_collection('ExtractedRelationships')
        
        assert edge_schema.name == 'ExtractedRelationships'
        assert edge_schema.count == 2
        assert 'ExtractedEntities' in edge_schema.from_collections
        assert 'ExtractedEntities' in edge_schema.to_collections
        assert 'Documents' in edge_schema.to_collections
        assert 'relationship' in edge_schema.fields
        assert '_from' in edge_schema.fields
        assert '_to' in edge_schema.fields
    
    def test_get_collection_info(self, mock_db, mock_collections, mock_entity_docs):
        """Test retrieving specific collection info."""
        # Setup mocks
        entity_col = Mock()
        entity_col.count.return_value = 2
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_entity_docs))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = entity_col
        mock_db.collections.return_value = [mock_collections[0]]  # Only ExtractedEntities
        
        inspector = SchemaInspector(mock_db)
        
        # Get existing collection
        col_info = inspector.get_collection_info('ExtractedEntities')
        assert col_info is not None
        assert col_info['name'] == 'ExtractedEntities'
        
        # Get non-existent collection
        missing_info = inspector.get_collection_info('NonExistent')
        assert missing_info is None
    
    def test_get_edge_info(self, mock_db, mock_collections, mock_relationship_edges):
        """Test retrieving specific edge collection info."""
        # Setup mocks
        edge_col = Mock()
        edge_col.count.return_value = 2
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter(mock_relationship_edges))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = edge_col
        mock_db.collections.return_value = [mock_collections[1]]  # Only ExtractedRelationships
        
        inspector = SchemaInspector(mock_db)
        
        # Get existing edge collection
        edge_info = inspector.get_edge_info('ExtractedRelationships')
        assert edge_info is not None
        assert edge_info['name'] == 'ExtractedRelationships'
        assert len(edge_info['from_collections']) > 0
        
        # Get non-existent edge collection
        missing_info = inspector.get_edge_info('NonExistent')
        assert missing_info is None
    
    def test_system_collections_ignored(self, mock_db, mock_collections):
        """Test that system collections (starting with _) are ignored."""
        # Setup mocks
        mock_db.collections.return_value = mock_collections
        
        entity_col = Mock()
        entity_col.count.return_value = 0
        
        mock_cursor = Mock()
        mock_cursor.__iter__ = Mock(return_value=iter([]))
        
        mock_aql = Mock()
        mock_aql.execute.return_value = mock_cursor
        mock_db.aql = mock_aql
        mock_db.collection.return_value = entity_col
        
        inspector = SchemaInspector(mock_db)
        schema = inspector.get_schema_map()
        
        # Verify _system collection is not in results
        all_names = [col['name'] for col in schema['collections']]
        all_names.extend([edge['name'] for edge in schema['edges']])
        assert '_system' not in all_names


class TestConvenienceFunction:
    """Test the convenience function."""
    
    def test_inspect_arango_schema(self, mock_db, mock_collections, tmp_path):
        """Test convenience function with file output."""
        # Setup mocks
        mock_db.collections.return_value = []
        mock_db.aql = Mock()
        
        output_file = tmp_path / "schema.json"
        
        schema = inspect_arango_schema(mock_db, output_file=str(output_file))
        
        assert schema is not None
        assert 'database_name' in schema
        assert output_file.exists()
