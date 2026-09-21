"""Explicit production configuration, with no unknown-name fallback."""

from dataclasses import asdict

import pytest

from npll.utils.config import FB15K_237_CONFIG, default_config, get_config


def test_odin_config_has_its_own_identity_without_retuning(capsys):
    config = get_config("OdinTriples")
    assert config.dataset_name == "OdinTriples"
    assert default_config is config
    assert config is not FB15K_237_CONFIG
    actual = asdict(config)
    previous = asdict(FB15K_237_CONFIG)
    del actual["dataset_name"]
    del previous["dataset_name"]
    assert actual == previous
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", ["ArangoDB_Triples", "OdinTriple", "unknown"])
def test_unknown_and_retired_config_names_fail(name):
    with pytest.raises(ValueError, match="Unknown NPLL dataset configuration"):
        get_config(name)


@pytest.mark.parametrize("name", ["FB15k-237", "WN18RR", "UMLS", "Kinship"])
def test_named_benchmark_configs_remain_available(name):
    assert get_config(name).dataset_name == name
