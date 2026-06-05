from __future__ import annotations

from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Source.dataLoader import load_data
from Source.analysis import (
    analyze_graphs,
    analyze_weighted_graphs,
    compare_degree_top_items,
    compare_weighted_degree_rankings,
    largest_component_centrality,
    top_central_items,
    top_pagerank_items,
    top_weighted_central_items,
)
from Source.categoryAnalysis import (
    analyze_component_category_alignment,
    build_category_edge_table,
    load_item_categories,
    summarize_component_alignment,
    summarize_component_alignment_size2plus,
    summarize_category_edges,
    summarize_category_pairs,
)
from Source.exportResults import export_result_tables
from Source.graphBuilder import (
    DEFAULT_K,
    DEFAULT_TAU,
    aggregate_similarity_across_cities,
    build_yearly_graphs,
    run_threshold_sensitivity,
    summarize_aggregated_similarity,
    summarize_graphs,
)
from Source.interactiveVisualization import create_interactive_visualizations
from Source.preProcessing import (
    add_window_month_index,
    assign_windows,
    complete_sequence_summary,
    filter_complete_sequences,
    filter_common_window_months,
    month_coverage,
    summarize_windows,
)
from Source.similarity import compute_citywise_similarity, summarize_similarity
from Source.temporalAnalysis import (
    persistent_edge_strength,
    summarize_temporal_edges,
    temporal_summary_table,
    top_temporal_edges,
    window_to_window_changes,
)
from Source.vectorBuilder import build_price_vectors, summarize_vectors
from Source.visualization import create_visualizations


DATASET_PATH = Path("Data/final_cpi_dataset.csv")
CATEGORY_PATH = Path("Data/item_categories.csv")
RESULTS_DIR = Path("Results")
VISUALS_DIR = RESULTS_DIR / "visuals"
INTERACTIVE_DIR = RESULTS_DIR / "interactive"
SENSITIVITY_TAU_VALUES = [0.3, 0.5, 0.7, 0.9]
SENSITIVITY_K_VALUES = [2, 4, 6, 8]


def main() -> None:
    raw_df = load_data(DATASET_PATH)
    windowed_df = assign_windows(raw_df)
    indexed_df = add_window_month_index(windowed_df)
    common_months_df = filter_common_window_months(indexed_df)
    complete_df = filter_complete_sequences(common_months_df)
    vectors_df = build_price_vectors(complete_df)
    similarity_df = compute_citywise_similarity(vectors_df)
    aggregated_df = aggregate_similarity_across_cities(similarity_df, tau=DEFAULT_TAU)
    graphs = build_yearly_graphs(aggregated_df, vectors_df, k_threshold=DEFAULT_K)
    analysis_results = analyze_graphs(graphs)
    top_items_df = top_central_items(analysis_results, top_n=5)
    pagerank_items_df = top_pagerank_items(graphs, weight_attribute="support_weight", top_n=5)
    largest_component_centrality_df = largest_component_centrality(graphs, top_n=5)
    weighted_results = {
        "support_cities": analyze_weighted_graphs(graphs, "support_weight"),
        "average_similarity": analyze_weighted_graphs(graphs, "similarity_weight"),
    }
    weighted_top_items_df = top_weighted_central_items(weighted_results, top_n=5)
    weighted_degree_comparison_df = compare_weighted_degree_rankings(weighted_top_items_df)
    temporal_edges_df = summarize_temporal_edges(graphs)
    temporal_summary_df = temporal_summary_table(temporal_edges_df)
    temporal_changes_df = window_to_window_changes(graphs)
    persistent_edge_strength_df = persistent_edge_strength(graphs)
    graph_summary_df = summarize_graphs(graphs)
    aggregated_summary_df = summarize_aggregated_similarity(aggregated_df)
    expected_items = sorted(vectors_df["Item"].unique().tolist())
    categories_df = load_item_categories(CATEGORY_PATH, expected_items)
    category_edges_df = build_category_edge_table(graphs, categories_df)
    category_summary_df = summarize_category_edges(category_edges_df)
    category_pairs_df = summarize_category_pairs(category_edges_df)
    component_alignment_df = analyze_component_category_alignment(graphs, categories_df)
    component_alignment_summary_df = summarize_component_alignment(component_alignment_df)
    component_alignment_size2plus_df = summarize_component_alignment_size2plus(component_alignment_df)
    percentage_similarity_df = compute_citywise_similarity(
        vectors_df,
        vector_column="PercentageChangeVector",
    )
    percentage_aggregated_df = aggregate_similarity_across_cities(
        percentage_similarity_df,
        tau=DEFAULT_TAU,
    )
    percentage_graphs = build_yearly_graphs(
        percentage_aggregated_df,
        vectors_df,
        k_threshold=DEFAULT_K,
    )
    percentage_top_items_df = top_central_items(analyze_graphs(percentage_graphs), top_n=5)
    percentage_change_comparison_df = compare_degree_top_items(
        top_items_df,
        percentage_top_items_df,
    )
    sensitivity_df = run_threshold_sensitivity(
        similarity_df,
        vectors_df,
        tau_values=SENSITIVITY_TAU_VALUES,
        k_values=SENSITIVITY_K_VALUES,
    )
    saved_files = export_result_tables(
        RESULTS_DIR,
        {
            "01_clean_analysis_data.csv": complete_df,
            "02_price_change_vectors.csv": vectors_df,
            "03_citywise_similarity.csv": similarity_df,
            "04_aggregated_similarity.csv": aggregated_df,
            "05_graph_summary.csv": graph_summary_df,
            "06_top_central_items.csv": top_items_df,
            "07_threshold_sensitivity.csv": sensitivity_df,
            "08_temporal_edges.csv": temporal_edges_df,
            "09_temporal_summary.csv": temporal_summary_df,
            "10_window_to_window_changes.csv": temporal_changes_df,
            "11_item_categories.csv": categories_df,
            "12_category_edges.csv": category_edges_df,
            "13_category_summary.csv": category_summary_df,
            "14_category_pair_summary.csv": category_pairs_df,
            "15_weighted_central_items.csv": weighted_top_items_df,
            "16_weighted_degree_comparison.csv": weighted_degree_comparison_df,
            "17_component_category_alignment.csv": component_alignment_df,
            "18_component_alignment_summary.csv": component_alignment_summary_df,
            "19_pagerank_items.csv": pagerank_items_df,
            "20_percentage_change_comparison.csv": percentage_change_comparison_df,
            "21_persistent_edge_strength.csv": persistent_edge_strength_df,
            "22_largest_component_centrality.csv": largest_component_centrality_df,
            "23_component_alignment_size2plus.csv": component_alignment_size2plus_df,
        },
    )
    saved_visuals = create_visualizations(
        VISUALS_DIR,
        graphs,
        graph_summary_df,
        top_items_df,
        sensitivity_df,
        weighted_top_items_df,
        categories_df,
        temporal_summary_df,
        category_summary_df,
        component_alignment_summary_df,
        aggregated_df,
    )
    saved_interactive_visuals = create_interactive_visualizations(
        INTERACTIVE_DIR,
        graphs,
        graph_summary_df,
        sensitivity_df,
        temporal_summary_df,
        category_summary_df,
        categories_df,
    )

    print("Step 1: Analysis-ready dataset")
    print()
    print("Window summary after applying the full March-to-February windows:")
    print(summarize_windows(indexed_df).to_string(index=False))
    print()

    print("Month coverage after keeping the full March-to-February windows:")
    print(month_coverage(common_months_df).to_string(index=False))
    print()

    print("Complete sequence summary:")
    print(complete_sequence_summary(common_months_df).to_string(index=False))
    print()

    print("Rows kept after filtering to complete 12-month item-city sequences:")
    print(len(complete_df))
    print()
    print("Window summary after filtering:")
    print(summarize_windows(complete_df).to_string(index=False))
    print()

    print("Step 2: Price-change vectors")
    print()
    print("Vector summary:")
    print(summarize_vectors(vectors_df).to_string(index=False))
    print()
    print("Sample vectors:")
    sample_columns = ["Window", "City", "Item", "PriceVector", "PriceChangeVector"]
    print(vectors_df.loc[:, sample_columns].head(6).to_string(index=False))
    print()

    print("Step 3: City-wise item similarity")
    print()
    print("Similarity summary:")
    print(summarize_similarity(similarity_df).to_string(index=False))
    print()
    print("Sample similarity rows:")
    print(similarity_df.head(10).to_string(index=False))
    print()

    print(f"Step 4: Aggregation across cities and yearly graphs (tau={DEFAULT_TAU}, K={DEFAULT_K})")
    print()
    print("Aggregated similarity summary:")
    print(aggregated_summary_df.to_string(index=False))
    print()
    print("Sample aggregated rows:")
    print(aggregated_df.head(10).to_string(index=False))
    print()
    print("Graph summary:")
    print(graph_summary_df.to_string(index=False))
    print()
    print("Top central items:")
    print(top_items_df.to_string(index=False))
    print()

    print("Weighted graph centrality comparison:")
    print(weighted_degree_comparison_df.to_string(index=False))
    print()
    print("PageRank top items:")
    print(pagerank_items_df.to_string(index=False))
    print()

    print("Step 5: Threshold sensitivity analysis")
    print()
    print("This table helps us choose thresholds that are neither too relaxed nor too strict.")
    print(sensitivity_df.to_string(index=False))
    print()

    print("Step 6: Temporal edge comparison")
    print()
    print("Temporal edge status summary:")
    print(temporal_summary_df.to_string(index=False))
    print()
    print("Window-to-window changes:")
    print(temporal_changes_df.to_string(index=False))
    print()
    print("Persistent edges:")
    persistent_edges = top_temporal_edges(temporal_edges_df, "Persistent", limit=10)
    if persistent_edges.empty:
        print("No edges were present in all three windows.")
    else:
        print(persistent_edges.to_string(index=False))
    print()
    print("Persistent edge strength:")
    print(persistent_edge_strength_df.to_string(index=False))
    print()
    print("Sample edges that appeared only in W3:")
    only_w3_edges = top_temporal_edges(temporal_edges_df, "Only W3", limit=10)
    if only_w3_edges.empty:
        print("No edges appeared only in W3.")
    else:
        print(only_w3_edges.to_string(index=False))
    print()

    print("Step 7: Exported result tables")
    print()
    for saved_file in saved_files:
        print(saved_file)
    print()

    print("Step 8: Exported visualizations")
    print()
    for saved_visual in saved_visuals:
        print(saved_visual)
    print()

    print("Step 8b: Exported interactive Plotly visualizations")
    print()
    for saved_interactive_visual in saved_interactive_visuals:
        print(saved_interactive_visual)
    print()

    print("Step 9: Category-based analysis")
    print()
    print("Within-category vs between-category edges:")
    print(category_summary_df.to_string(index=False))
    print()
    print("Top category-pair relationships:")
    print(category_pairs_df.groupby("Window").head(5).to_string(index=False))
    print()
    print("Component/category alignment summary:")
    print(component_alignment_summary_df.to_string(index=False))
    print()
    print("Component/category alignment summary for components with at least 2 items:")
    print(component_alignment_size2plus_df.to_string(index=False))
    print()
    print("Absolute-change vs percentage-change degree comparison:")
    print(percentage_change_comparison_df.to_string(index=False))
    print()
    print("Largest components and their main categories:")
    print(component_alignment_df.groupby("Window").head(3).loc[:, [
        "Window",
        "Component",
        "ComponentSize",
        "CategoryCount",
        "MainCategory",
        "PurityPercent",
        "Alignment",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
