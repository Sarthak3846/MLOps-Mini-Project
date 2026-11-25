import hashlib
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mp046.cleaning import clean_data
from mp046.config import ROOT, features, load_params
from mp046.modeling import build_pipeline, candidates


def main():
    params = load_params()
    feats, target = features(params), params["target"]

    train_file = ROOT / params["train_path"]
    data_sha256 = hashlib.sha256(train_file.read_bytes()).hexdigest()

    df = clean_data(pd.read_csv(train_file), params)
    split = int(len(df) * (1 - params["val_fraction"]))  # time-based split
    tr, va = df.iloc[:split], df.iloc[split:]

    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment(params["experiment_name"])

    results = []
    for name, model_params, model in candidates(params):
        pipe = build_pipeline(model, params)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("features", ",".join(feats))
            mlflow.log_param("n_rows_clean", len(df))
            mlflow.log_params(model_params)

            pipe.fit(tr[feats], tr[target])
            mae_val = mean_absolute_error(va[target], pipe.predict(va[feats]))
            mlflow.log_metric("mae_val", mae_val)

            final = clone(pipe).fit(df[feats], df[target])  # v1 = all cleaned train
            mlflow.sklearn.log_model(final, name="model", serialization_format="cloudpickle")
            results.append((mae_val, run.info.run_id, final, name))
            print(f"{name:18s} mae_val={mae_val:.3f} run_id={run.info.run_id}")

    mae_val, best_id, best_model, best_name = min(results, key=lambda r: r[0])
    name = params["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_id}/model", name)
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)

    out = ROOT / params["model_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, out)

    print(f"\nbest={best_name} mae_val={mae_val:.3f}")
    print(f"experiment_name: {params['experiment_name']}")
    print(f"best_run_id: {best_id}")
    print(f"registered_model_name: {name}")
    print(f"champion_version: {mv.version}")


if __name__ == "__main__":
    main()
