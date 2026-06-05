from __future__ import annotations

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.analysis import analyze_graphs, top_central_items, top_pagerank_items
from Source.dataLoader import load_data
from Source.graphBuilder import (
    DEFAULT_K,
    DEFAULT_TAU,
    aggregate_similarity_across_cities,
    build_yearly_graphs,
    summarize_graphs,
)
from Source.preProcessing import (
    add_window_month_index,
    assign_windows,
    filter_common_window_months,
    filter_complete_sequences,
)
from Source.similarity import compute_citywise_similarity
from Source.vectorBuilder import build_price_vectors


DATASET_PATH = Path("Data/final_cpi_dataset.csv")


def main() -> None:
    print("CPI Graph Project - Code Defense Walkthrough")
    print()

    raw_df = load_data(DATASET_PATH)
    print(f"1. Loaded dataset: {len(raw_df)} clean price records")
    print(raw_df.head(3).to_string(index=False))
    print()

    windowed_df = assign_windows(raw_df)
    indexed_df = add_window_month_index(windowed_df)
    complete_df = filter_complete_sequences(filter_common_window_months(indexed_df))
    print("2. Created W1, W2, W3 windows and kept complete item-city sequences")
    print(complete_df[["Window", "Year", "Month", "City", "Item", "Price"]].head(3).to_string(index=False))
    print()

    vectors_df = build_price_vectors(complete_df)
    sample_vector = vectors_df.iloc[0]
    print("3. Converted each item-city-window record into price-change vectors")
    print(f"Item: {sample_vector['Item']}")
    print(f"Prices: {sample_vector['PriceVector']}")
    print(f"Price changes: {sample_vector['PriceChangeVector']}")
    print()

    similarity_df = compute_citywise_similarity(vectors_df)
    print("4. Compared every item pair inside each city using cosine similarity")
    print(similarity_df.head(5).to_string(index=False))
    print()

    aggregated_df = aggregate_similarity_across_cities(similarity_df, tau=DEFAULT_TAU)
    print(f"5. Counted city support using tau = {DEFAULT_TAU}")
    print(aggregated_df.head(5).to_string(index=False))
    print()

    graphs = build_yearly_graphs(aggregated_df, vectors_df, k_threshold=DEFAULT_K)
    print(f"6. Built graphs using K = {DEFAULT_K}")
    print("Edge rule: if SupportCities >= K, we connect the two items.")
    print(summarize_graphs(graphs).to_string(index=False))
    print()

    centrality_results = analyze_graphs(graphs)
    top_items_df = top_central_items(centrality_results, top_n=3)
    print("7. Calculated degree, closeness, and betweenness centrality")
    print(top_items_df.to_string(index=False))
    print()

    pagerank_df = top_pagerank_items(graphs, top_n=3)
    print("8. Calculated PageRank")
    print(pagerank_df.to_string(index=False))
    print()

    print("Defense summary:")
    print("- Nodes are CPI items.")
    print("- Edges mean similar price movement in enough cities.")
    print("- tau checks similarity inside one city.")
    print("- K checks how many cities agree.")
    print("- Centrality and PageRank identify important items in the graph.")


if __name__ == "__main__":
    main()
