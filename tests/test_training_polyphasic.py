import torch
from chaoscontrol.config import ChaosControlConfig


def test_polyphasic_config_fields():
    cfg = ChaosControlConfig(
        data_path="dummy",
        polyphasic_enabled=True,
        polyphasic_n_partitions=4,
        polyphasic_k_awake=3,
        polyphasic_topology="bucket_owned",
        polyphasic_swap_interval=256,
    )
    assert cfg.polyphasic_enabled is True
    assert cfg.polyphasic_n_partitions == 4
    assert cfg.polyphasic_k_awake == 3
    assert cfg.polyphasic_topology == "bucket_owned"


def test_polyphasic_config_defaults():
    cfg = ChaosControlConfig(data_path="dummy")
    assert cfg.polyphasic_enabled is False
    assert cfg.polyphasic_topology == "slot_striped"
    assert cfg.polyphasic_n_partitions == 4
    assert cfg.polyphasic_k_awake == 3
    assert cfg.polyphasic_swap_interval == 256
