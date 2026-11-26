import pandas as pd

from mp046.cleaning import fix_discount_units, fix_target_typos, normalise_shop_type


def test_shop_type_variants_collapse_to_three_levels():
    s = pd.Series(["chain", " Chain ", "CHA.", "independent", "ind.", "hospital store", "hos.", "hospital_store"])
    assert set(normalise_shop_type(s)) == {"chain", "independent", "hospital_store"}


def test_cantonment_discount_converted_back_to_percent():
    df = pd.DataFrame(
        {"locality": ["Cantonment", "Cantonment", "Old Town", "Old Town"], "discount_pct": [0.1, 0.2, 10.0, 20.0]}
    )
    out = fix_discount_units(df)
    assert out.loc[out.locality == "Cantonment", "discount_pct"].tolist() == [10.0, 20.0]
    assert out.loc[out.locality == "Old Town", "discount_pct"].tolist() == [10.0, 20.0]


def test_target_typos_divided_by_ten():
    df = pd.DataFrame({"units_sold": [300, 4500, 250]})
    assert fix_target_typos(df, "units_sold", 2000)["units_sold"].tolist() == [300, 450, 250]
