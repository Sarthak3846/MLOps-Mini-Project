import mlflow
import mlflow.sklearn
import pandas as pd

from mp046.cleaning import prepare_features
from mp046.config import ROOT, features, load_params


def main():
    params = load_params()
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    model = mlflow.sklearn.load_model(f"models:/{params['registered_model_name']}@champion")

    raw = pd.read_csv(ROOT / params["holdout_path"])
    X = prepare_features(raw)[features(params)]
    out = pd.DataFrame({"record_id": raw["record_id"], "prediction": model.predict(X)})

    path = ROOT / params["predictions_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {len(out)} rows to {path}")


if __name__ == "__main__":
    main()
