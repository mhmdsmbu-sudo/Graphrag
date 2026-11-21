import os
import pandas as pd
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH

ACRONYM_FIX = {"Ceo": "CEO", "Cto": "CTO", "Coo": "COO", "Cpo": "CPO", "Cfo": "CFO", "Cmo": "CMO"}
BOOL_MAP = {"yes": True, "y": True, "true": True, "1": True, "no": False, "n": False, "false": False, "0": False}


def strip_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from all string columns."""
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notna(x) else x)
    return df


def to_nullable_int(series: pd.Series) -> pd.Series:
    """Convert float-like year columns (e.g., 2016.0) to nullable integers (Int64)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def map_canonical(series: pd.Series, lookup: dict) -> pd.Series:
    """Standardize entity names using a case-insensitive canonical lookup map."""
    return series.apply(lambda x: lookup.get(str(x).strip().lower(), str(x).strip()) if pd.notna(x) else x)


def clean_nodes(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean all 5 node sheets: whitespace, casing, numeric formats, and duplicates."""
    founders = strip_text_columns(raw["Founders"])
    founders["role"] = founders["role"].str.title().replace(ACRONYM_FIX)
    founders["born"] = to_nullable_int(founders["born"])
    founders = founders.drop_duplicates(subset=["name"]).reset_index(drop=True)

    companies = strip_text_columns(raw["Companies"])
    companies["sector"] = companies["sector"].replace({"ai": "AI"})
    companies["founded"] = to_nullable_int(companies["founded"])
    companies["valuation_billion"] = pd.to_numeric(companies["valuation_billion"], errors="coerce")
    companies = companies.drop_duplicates(subset=["name"]).reset_index(drop=True)

    investors = strip_text_columns(raw["Investors"])
    investors["founded"] = to_nullable_int(investors["founded"])
    investors = investors.drop_duplicates(subset=["name"]).reset_index(drop=True)

    products = strip_text_columns(raw["Products"])
    products["launch_year"] = to_nullable_int(products["launch_year"])
    products = products.drop_duplicates(subset=["name"]).reset_index(drop=True)

    awards = strip_text_columns(raw["Awards"])
    awards["year"] = to_nullable_int(awards["year"])
    awards = awards.drop_duplicates(subset=["name"]).reset_index(drop=True)

    return {
        "Founders": founders,
        "Companies": companies,
        "Investors": investors,
        "Products": products,
        "Awards": awards,
    }


def clean_relationships(raw: dict[str, pd.DataFrame], nodes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean all 6 relationship sheets and drop dangling references."""
    lookups = {k: {n.lower(): n for n in df["name"]} for k, df in nodes.items()}
    valid = {k: set(df["name"]) for k, df in nodes.items()}

    # Founded
    founded = strip_text_columns(raw["Founded"])
    founded["founder_name"] = map_canonical(founded["founder_name"], lookups["Founders"])
    founded["company_name"] = map_canonical(founded["company_name"], lookups["Companies"])
    founded["year"] = to_nullable_int(founded["year"])
    founded = founded[
        founded["founder_name"].isin(valid["Founders"]) & founded["company_name"].isin(valid["Companies"])
    ].drop_duplicates().reset_index(drop=True)

    # WorksAt
    worksat = strip_text_columns(raw["WorksAt"])
    worksat["person_name"] = map_canonical(worksat["person_name"], lookups["Founders"])
    worksat["company_name"] = map_canonical(worksat["company_name"], lookups["Companies"])
    worksat["current"] = worksat["current"].apply(lambda x: BOOL_MAP.get(str(x).strip().lower(), x))
    worksat = worksat[
        worksat["person_name"].isin(valid["Founders"]) & worksat["company_name"].isin(valid["Companies"])
    ].drop_duplicates().reset_index(drop=True)

    # InvestedIn
    investedin = strip_text_columns(raw["InvestedIn"])
    investedin["investor_name"] = map_canonical(investedin["investor_name"], lookups["Investors"])
    investedin["company_name"] = map_canonical(investedin["company_name"], lookups["Companies"])
    investedin["amount_million"] = pd.to_numeric(investedin["amount_million"], errors="coerce")
    investedin["year"] = to_nullable_int(investedin["year"])
    investedin = investedin[
        investedin["investor_name"].isin(valid["Investors"]) & investedin["company_name"].isin(valid["Companies"])
    ].drop_duplicates().reset_index(drop=True)

    # Acquired
    acquired = strip_text_columns(raw["Acquired"])
    acquired["acquirer_company"] = map_canonical(acquired["acquirer_company"], lookups["Companies"])
    acquired["target_company"] = map_canonical(acquired["target_company"], lookups["Companies"])
    acquired["year"] = to_nullable_int(acquired["year"])
    acquired["amount_million"] = pd.to_numeric(acquired["amount_million"], errors="coerce")
    acquired = acquired[
        acquired["acquirer_company"].isin(valid["Companies"]) & acquired["target_company"].isin(valid["Companies"])
    ].drop_duplicates().reset_index(drop=True)

    # Develops
    develops = strip_text_columns(raw["Develops"])
    develops["company_name"] = map_canonical(develops["company_name"], lookups["Companies"])
    develops["product_name"] = map_canonical(develops["product_name"], lookups["Products"])
    develops = develops[
        develops["company_name"].isin(valid["Companies"]) & develops["product_name"].isin(valid["Products"])
    ].drop_duplicates().reset_index(drop=True)

    # Won (polymorphic: Company OR Person)
    won = strip_text_columns(raw["Won"])
    won["entity_type"] = won["entity_type"].str.capitalize()
    won["award_name"] = map_canonical(won["award_name"], lookups["Awards"])
    won["entity_name"] = won.apply(
        lambda r: lookups["Companies"].get(r["entity_name"].lower(), r["entity_name"])
        if r["entity_type"] == "Company"
        else lookups["Founders"].get(r["entity_name"].lower(), r["entity_name"]),
        axis=1,
    )
    valid_won = won["award_name"].isin(valid["Awards"]) & (
        ((won["entity_type"] == "Company") & won["entity_name"].isin(valid["Companies"]))
        | ((won["entity_type"] == "Person") & won["entity_name"].isin(valid["Founders"]))
    )
    won = won[valid_won].drop_duplicates().reset_index(drop=True)

    return {
        "Founded": founded,
        "WorksAt": worksat,
        "InvestedIn": investedin,
        "Acquired": acquired,
        "Develops": develops,
        "Won": won,
    }


def clean_dataset():
    """Run full cleaning pipeline and save a single cleaned Excel workbook."""
    xls = pd.ExcelFile(RAW_DATA_PATH)
    raw = {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names if s != "README"}

    nodes = clean_nodes(raw)
    rels = clean_relationships(raw, nodes)
    cleaned = {**nodes, **rels}

    os.makedirs(os.path.dirname(CLEAN_DATA_PATH), exist_ok=True)
    with pd.ExcelWriter(CLEAN_DATA_PATH, engine="openpyxl") as writer:
        print("=== Data Cleaning Summary (Raw -> Clean) ===")
        for sheet_name, df in cleaned.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  {sheet_name:12s}: {len(raw[sheet_name]):2d} -> {len(df):2d}")

    print(f"\nSaved cleaned workbook to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    clean_dataset()
