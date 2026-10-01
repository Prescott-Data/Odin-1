from unittest.mock import Mock
import pytest
from odin.backends.arango import ArangoGraphConfig
from odin.backends.base import BackendConfigurationError, BackendIOError
from retrieval.writers.arango_writer import ArangoWriter


def test_writer_preserves_full_ids_exact_relations_and_mapped_fields():
    db = Mock()
    graph = ArangoGraphConfig("Records", "Assertions", "predicate")
    writer = ArangoWriter(db, graph, confidence_field="certainty", metadata_field="evidence")
    assert writer.maybe_write_link("Other/a", "Submitted By", "Records/a", 0.9,
                                   {"embedding": [1], "text": "complete"})
    db.collection.assert_called_once_with("Assertions")
    document = db.collection.return_value.insert.call_args.args[0]
    assert document == {"_from": "Other/a", "_to": "Records/a", "predicate": "Submitted By",
                        "certainty": 0.9, "evidence": {"embedding": [1], "text": "complete"}}
    assert not writer.maybe_write_link("Other/a", "r", "Records/a", 0.2)
    with pytest.raises(BackendConfigurationError, match="full Arango"):
        writer.maybe_write_link("a", "r", "Records/a", 0.9)
    db.collection.return_value.insert.side_effect = OSError("offline")
    with pytest.raises(BackendIOError):
        writer.maybe_write_link("Other/a", "r", "Records/a", 0.9)
