from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    expected_columns = {"Year", "Month", "City", "Item", "Price"}
    missing_columns = expected_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    cleaned = df.loc[:, ["Year", "Month", "City", "Item", "Price"]].copy()
    cleaned["Year"] = pd.to_numeric(cleaned["Year"], errors="coerce").astype("Int64")
    cleaned["Month"] = pd.to_numeric(cleaned["Month"], errors="coerce").astype("Int64")
    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")

    # Drop incomplete rows before converting text fields so missing values stay missing.
    cleaned = cleaned.dropna(subset=["Year", "Month", "City", "Item", "Price"]).reset_index(drop=True)
    cleaned["City"] = cleaned["City"].astype("string").str.strip()
    cleaned["Item"] = cleaned["Item"].astype("string").str.strip()
    cleaned = cleaned.loc[cleaned["City"].ne("") & cleaned["Item"].ne("")].reset_index(drop=True)
    return cleaned
