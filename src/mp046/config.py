from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent


def load_params(path=None):
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def features(params):
    return params["numeric_features"] + params["categorical_features"]
