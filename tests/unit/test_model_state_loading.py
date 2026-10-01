import copy
from unittest.mock import patch
import pytest
from npll.bootstrap import KnowledgeBootstrapper
from npll.utils.config import get_config
from odin.backends.base import CorruptModelError, MODEL_KEY, StoredModel
from retrieval.confidence import NPLLConfidence
from tests.utils.backend_fakes import MemorySource, MemoryStore
from tests.unit.test_training_telemetry import make_training_result


def trained_fixture():
    source = MemorySource([('Nodes/a', 'arbitrary', 'Nodes/b'), ('Nodes/b', 'other', 'Nodes/c')])
    store = MemoryStore()
    config = get_config('OdinTriples')
    config.scorer_epochs = 2
    with patch('npll.bootstrap.get_config', return_value=config), patch('npll.bootstrap.create_trainer') as trainer:
        trainer.return_value.train.return_value = make_training_result()
        result = KnowledgeBootstrapper(source, store).ensure_model_ready()
    return source, store, config, result


def test_reload_loads_scorer_weights_without_training_or_runtime_invalidation():
    source, store, config, trained = trained_fixture()
    document = copy.deepcopy(store.docs[MODEL_KEY].document)
    document['inference_state']['config']['device'] = 'cuda'
    document['inference_state']['scorer_training']['torch_version'] = 'other-runtime'
    document['inference_state']['scorer_training']['loss_history'][0] += 1e-8
    store.docs[MODEL_KEY] = StoredModel(document, '1')
    with patch('npll.bootstrap.get_config', return_value=config), \
         patch('npll.bootstrap.create_snapshot_trained_model', side_effect=AssertionError('startup trained')):
        loaded = KnowledgeBootstrapper(source, store).ensure_model_ready()
    assert loaded.source == 'cached_weights'
    assert store.saves == 1
    assert NPLLConfidence(loaded.model).confidence_batch(source.triples) == \
        NPLLConfidence(trained.model).confidence_batch(source.triples)


@pytest.mark.parametrize('change', [
    lambda d: d.update(version='6.0'),
    lambda d: d.update(version='future-version'),
    lambda d: d['inference_state']['config'].pop('scorer_negatives_per_side'),
    lambda d: d['inference_state']['config'].update(future_setting=42),
    lambda d: d['inference_state']['config'].update(scorer_learning_rate=0.05),
])
def test_obsolete_artifact_is_retrained_and_replaced_with_revision(change):
    source, store, config, _ = trained_fixture()
    document = copy.deepcopy(store.docs[MODEL_KEY].document)
    change(document)
    store.docs[MODEL_KEY] = StoredModel(document, '1')
    with patch('npll.bootstrap.get_config', return_value=config), patch('npll.bootstrap.create_trainer') as trainer:
        trainer.return_value.train.return_value = make_training_result()
        assert KnowledgeBootstrapper(source, store).ensure_model_ready().source == 'trained'
    assert store.saves == 2


def test_damaged_scorer_blob_is_corruption_not_staleness():
    source, store, config, _ = trained_fixture()
    document = copy.deepcopy(store.docs[MODEL_KEY].document)
    document['inference_state']['scorer_state']['sha256'] = '0' * 64
    store.docs[MODEL_KEY] = StoredModel(document, '1')
    with patch('npll.bootstrap.get_config', return_value=config), pytest.raises(CorruptModelError):
        KnowledgeBootstrapper(source, store).ensure_model_ready()
    assert store.saves == 1
