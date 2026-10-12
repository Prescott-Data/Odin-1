"""Odin preserves application documents; model-input filtering belongs to callers."""
from copy import deepcopy
import pytest
from retrieval.adapters_arango import ArangoCommunityAccessor
from tests.utils.backend_fakes import FakeArango


def document():
    return {'_id': 'Sources/a', 'embedding': [float(i) for i in range(3001)],
            'nested': {'npll_embedding': [0.1, 0.2], 'transe_embedding': [0.3, 0.4],
                       'embedding_source': 'complete source text'},
            'odin_excluded_vector_fields': 'application-owned metadata',
            'text': 'tail evidence'}


def test_get_node_preserves_all_fields_including_vectors_and_application_metadata():
    db = FakeArango()
    raw = document()
    db.aql.execute = lambda *args, **kwargs: iter([deepcopy(raw)])
    accessor = ArangoCommunityAccessor(db, community_id='global', nodes_collection='Sources',
                                      edges_collection='Edges', relation_property='predicate')
    assert accessor.get_node('Sources/a') == raw


@pytest.mark.parametrize('helper', ['get_document_content', 'get_entity_sources', 'search_content'])
def test_content_and_provenance_helpers_preserve_complete_vectors(helper):
    db = FakeArango()
    raw = document()
    row = {'source_id': 'Sources/a', 'source_type': 'Sources', 'document': raw,
           'edge': {'_id': 'Provenance/a', 'embedding': [1, 2], 'predicate': 'Sourced From'}}
    db.aql.execute = lambda *args, **kwargs: iter([deepcopy(row)])
    if helper == 'get_document_content':
        result = ArangoCommunityAccessor.get_document_content(
            db, 'Sources/a', text_collection='Sources', table_collection='Tables',
            image_collection='Images', document_collection='Documents')
        assert result == row
    elif helper == 'get_entity_sources':
        assert ArangoCommunityAccessor.get_entity_sources(
            db, 'Entities/a', extracted_from_collection='Provenance') == [row]
    else:
        assert ArangoCommunityAccessor.search_content(
            db, 'tail', content_types=['Sources'], text_collection='Sources',
            table_collection='Tables', image_collection='Images',
            text_search_fields=['text'], table_search_fields=['text'], image_search_fields=['text']) == [row]
