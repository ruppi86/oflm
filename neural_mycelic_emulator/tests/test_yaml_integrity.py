import yaml
from pathlib import Path

import pytest

CONFIG_PATH = Path(__file__).resolve().parents[2] / "models" / "emulator_parameters.yml"

cfg_all = yaml.safe_load(CONFIG_PATH.read_text())

@pytest.mark.parametrize("tag, info", cfg_all["models"].items())
def test_dataset_exists(tag, info):
    path = Path(info.get("dataset", ""))
    assert path.exists(), f"Dataset path missing for {tag}: {path}"

@pytest.mark.parametrize("tag, info", cfg_all["models"].items())
def test_required_keys(tag, info):
    required = {"hidden_dim", "num_layers", "epochs"}
    missing = required - info.keys()
    assert not missing, f"{tag} missing keys: {missing}" 