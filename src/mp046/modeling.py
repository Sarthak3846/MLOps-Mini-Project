from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model, params):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), params["numeric_features"]),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                params["categorical_features"],
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def candidates(params):
    rs = params["random_state"]
    return [
        ("ridge", {"model": "Ridge", "alpha": 1.0}, Ridge(alpha=1.0)),
        (
            "random_forest",
            {"model": "RandomForestRegressor", "n_estimators": 300, "max_depth": 12},
            RandomForestRegressor(n_estimators=300, max_depth=12, random_state=rs, n_jobs=-1),
        ),
        (
            "gradient_boosting",
            {"model": "GradientBoostingRegressor", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
            GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=rs),
        ),
    ]
