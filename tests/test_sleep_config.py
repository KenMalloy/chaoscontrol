from chaoscontrol.config import ChaosControlConfig

def test_sleep_config_defaults():
    cfg = ChaosControlConfig(data_path="/tmp")
    assert cfg.sleep_enabled is False
    assert cfg.sleep_stages == "full_cycle"
    assert cfg.sleep_budget == 128
    assert cfg.sleep_n2_budget == 64
    assert cfg.sleep_rem_budget == 64
    assert cfg.sleep_adaptive_fatigue is False
