"""Explicit production configuration, with no unknown-name fallback."""

from dataclasses import asdict

import pytest

from npll.utils.config import FB15K_237_CONFIG, default_config, get_config


def test_odin_config_has_a_compact_explicit_production_profile(capsys):
    config = get_config("OdinTriples")
    assert config.dataset_name == "OdinTriples"
    assert default_config is config
    assert config is not FB15K_237_CONFIG
    assert config.entity_embedding_dim == 32
    assert config.relation_embedding_dim == 32
    assert config.rule_embedding_dim == 64
    assert config.scoring_hidden_dim == 64
    assert asdict(config) != asdict(FB15K_237_CONFIG)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", ["ArangoDB_Triples", "OdinTriple", "unknown"])
def test_unknown_and_retired_config_names_fail(name):
    with pytest.raises(ValueError, match="Unknown NPLL dataset configuration"):
        get_config(name)


@pytest.mark.parametrize("name", ["FB15k-237", "WN18RR", "UMLS", "Kinship"])
def test_named_benchmark_configs_remain_available(name):
    assert get_config(name).dataset_name == name
