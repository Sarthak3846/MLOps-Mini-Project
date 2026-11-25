import pandas as pd

SHOP_TYPE_MAP = {
    "ind.": "independent",
    "cha.": "chain",
    "hos.": "hospital_store",
    "hospital store": "hospital_store",
}


def normalise_shop_type(series):
    s = series.astype(str).str.strip().str.lower()
    return s.replace(SHOP_TYPE_MAP)


def fix_discount_units(df):
    """Cantonment typed discount_pct as a fraction (x0.01); convert back to percent."""
    df = df.copy()
    means = df.groupby("locality")["discount_pct"].mean()
    for locality, mean in means.items():
        if mean < 1:
            df.loc[df["locality"] == locality, "discount_pct"] *= 100
    df["discount_pct"] = df["discount_pct"].round(2)
    return df


def fix_target_typos(df, target, threshold):
    """Targets typed 10x too big: divide by 10."""
    df = df.copy()
    typo = df[target] > threshold
    df.loc[typo, target] = df.loc[typo, target] / 10
    return df


def clean_data(df, params, has_target=True):
    df = df.drop_duplicates().copy()
    df["shop_type"] = normalise_shop_type(df["shop_type"])
    df = fix_discount_units(df)
    if has_target:
        df = fix_target_typos(df, params["target"], params["target_typo_threshold"])
    return df.sort_values(["date", "record_id"]).reset_index(drop=True)


def prepare_features(df):
    """Row-preserving feature fixes (no dedupe, no sorting) for scoring new data."""
    df = df.copy()
    df["shop_type"] = normalise_shop_type(df["shop_type"])
    return fix_discount_units(df)
