"""
Standalone demo: run the full KG debugger pipeline on the built-in sample KG.
Usage: python demo_kg.py
"""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.curvature import get_node_curvature

from kg_debugger import (
    create_sample_kg,
    get_kg_stats,
    compute_kg_curvature,
    identify_bridge_entities,
    detect_reasoning_bottlenecks,
    trace_reasoning_path,
    generate_multi_hop_questions,
    evaluate_reachability,
    suggest_new_connections,
    auto_fix,
    compare_before_after,
    generate_text_report,
    generate_markdown_report,
    plot_kg_overview,
    plot_bottleneck_subgraph,
    plot_reasoning_path,
    plot_before_after_comparison,
    plot_relation_curvature_heatmap,
    plot_health_dashboard,
    compute_relation_curvature_stats,
    generate_diagnostic_summary,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _node_curvature_from_edges(G, G_curved, edge_curvatures):
    nc = get_node_curvature(G_curved) if G_curved.number_of_nodes() else {}
    return {n: nc.get(n, 0.0) for n in G.nodes()}


def main():
    print("=== Knowledge Graph Debugger Demo ===\n")

    # 1. Create sample KG
    print("1. Creating sample tech company KG...")
    G = create_sample_kg()
    print("   Done.\n")

    # 2. KG stats
    print("2. KG statistics:")
    get_kg_stats(G)

    # 3. Compute curvature
    print("3. Computing curvature (alpha=0.5)...")
    G_curved, edge_curvatures = compute_kg_curvature(G, alpha=0.5)
    node_curvature = _node_curvature_from_edges(G, G_curved, edge_curvatures)
    print("   Done.\n")

    # 4. Top 10 bottlenecks
    print("4. Top 10 bottleneck edges:")
    bottlenecks_df = detect_reasoning_bottlenecks(G, edge_curvatures, node_curvature, top_k=10)
    if bottlenecks_df.empty:
        print("   (none)")
    else:
        for i, row in bottlenecks_df.iterrows():
            print(f"   {row['source_label']} --[{row['relation']}]--> {row['target_label']}  curvature={row['curvature']:.4f}  impact={row['impact_score']:.4f}")
    print()

    # 5. Bridge entities
    print("5. Bridge entities (at risk):")
    bridge_entities = identify_bridge_entities(G, node_curvature, threshold=-0.1)
    if not bridge_entities:
        print("   (none)")
    else:
        for node_id, curv, bet, label, etype in bridge_entities[:15]:
            print(f"   {label} ({etype}): curvature={curv:.4f}, betweenness={bet:.4f}")
    print()

    # 6. Generate 20 multi-hop questions
    print("6. Generating 20 multi-hop questions...")
    questions = generate_multi_hop_questions(G, num_questions=20, min_hops=2, max_hops=4)
    print(f"   Generated {len(questions)} questions.\n")

    # 7. Evaluate reachability
    print("7. Evaluating reasoning reachability...")
    overall, per_scores = evaluate_reachability(G, questions, edge_curvatures)
    print(f"   Overall reachability score: {overall:.4f}\n")

    # 8. Suggest fixes
    print("8. Suggesting fixes...")
    suggestions_df = suggest_new_connections(G, edge_curvatures, node_curvature, max_suggestions=20)
    print(f"   Generated {len(suggestions_df)} suggestions.\n")

    # 9. Auto-fix
    print("9. Applying auto-fix (budget=20)...")
    G_rewired, report = auto_fix(G, edge_curvatures, node_curvature, budget=20)
    print(f"   Health before: {report.get('health_before', 0):.1f}, after: {report.get('health_after', 0):.1f}, improvement: {report.get('improvement', 0):.2f}")
    print(f"   Edges added: {report.get('num_edges_added', 0)}\n")

    # 10. Re-evaluate reasoning
    print("10. Re-evaluating reachability after rewiring...")
    compare_df = compare_before_after(G, G_rewired, questions, edge_curvatures)
    improved = compare_df["improved"].sum() if not compare_df.empty else 0
    print(f"   Questions that became more reachable: {improved} / {len(compare_df)}\n")

    # 11. Generate report
    print("11. Generating diagnostic report...")
    diagnostic = generate_diagnostic_summary(G, edge_curvatures, node_curvature)
    reasoning_eval = {"overall_reachability": overall, "improvement": report.get("improvement", 0)}
    generate_text_report(diagnostic, suggestions_df, reasoning_eval)
    generate_markdown_report(diagnostic, suggestions_df, reasoning_eval, "figures")
    print("   Text and Markdown reports written to results/.\n")

    # 12. Save all figures
    print("12. Saving figures...")
    plot_kg_overview(G, node_curvature, "KG Overview", edge_curvatures=edge_curvatures)
    if not bottlenecks_df.empty:
        u, v = int(bottlenecks_df.iloc[0]["source"]), int(bottlenecks_df.iloc[0]["target"])
        plot_bottleneck_subgraph(G, [(u, v)], edge_curvatures, "Top bottleneck subgraph")
    if questions:
        path_data = trace_reasoning_path(G, questions[0]["start"], questions[0]["end"], edge_curvatures)
        if path_data:
            plot_reasoning_path(G, path_data, "Sample reasoning path")
    rel_stats = compute_relation_curvature_stats(G_curved, edge_curvatures)
    if not rel_stats.empty:
        plot_relation_curvature_heatmap(rel_stats)
    plot_health_dashboard(diagnostic)
    try:
        G_curved_r, edge_curv_r = compute_kg_curvature(G_rewired, alpha=0.5)
        node_curv_r = _node_curvature_from_edges(G_rewired, G_curved_r, edge_curv_r)
        plot_before_after_comparison(G, G_rewired, node_curvature, node_curv_r)
    except Exception as e:
        logger.warning("Could not plot before/after: %s", e)
    print("   Figures saved to figures/.\n")

    print("Done! Check results/ and figures/.")


if __name__ == "__main__":
    main()
