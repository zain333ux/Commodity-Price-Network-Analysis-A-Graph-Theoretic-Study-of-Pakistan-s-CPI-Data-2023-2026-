from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("Results")
DEFAULT_K = 5


def main() -> None:
    checks = [
        _check_vector_lengths(),
        _check_graph_nodes(),
        _check_edge_support(),
        _check_cosine_range(),
    ]

    print("Validation checks")
    print()
    for passed, message in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {message}")

    if not all(passed for passed, _ in checks):
        raise SystemExit(1)


def _check_vector_lengths() -> tuple[bool, str]:
    vectors_df = pd.read_csv(RESULTS_DIR / "02_price_change_vectors.csv")
    valid = bool((vectors_df["VectorLength"] == 11).all())
    return valid, "every price-change vector has length 11"


def _check_graph_nodes() -> tuple[bool, str]:
    graph_summary_df = pd.read_csv(RESULTS_DIR / "05_graph_summary.csv")
    valid = bool((graph_summary_df["Nodes"] == 51).all())
    return valid, "every yearly graph has 51 nodes"


def _check_edge_support() -> tuple[bool, str]:
    edges_df = pd.read_csv(RESULTS_DIR / "12_category_edges.csv")
    valid = bool((edges_df["SupportCities"] >= DEFAULT_K).all())
    return valid, f"every exported graph edge has support cities >= {DEFAULT_K}"


def _check_cosine_range() -> tuple[bool, str]:
    similarity_df = pd.read_csv(RESULTS_DIR / "03_citywise_similarity.csv")
    valid = bool(similarity_df["CosineSimilarity"].between(-1, 1).all())
    return valid, "every cosine similarity value is between -1 and 1"


if __name__ == "__main__":
    main()
