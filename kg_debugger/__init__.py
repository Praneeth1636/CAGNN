"""
Knowledge Graph Debugger: find and fix weak connections in knowledge graphs
using curvature analysis for multi-hop reasoning.
"""

from .kg_loader import (
    load_from_triples,
    load_from_json,
    load_freebase_subset,
    load_wn18rr_subset,
    create_sample_kg,
    get_kg_stats,
    to_undirected,
)
from .kg_curvature import (
    compute_kg_curvature,
    compute_relation_curvature_stats,
    compute_entity_type_curvature,
    identify_bridge_entities,
)
from .bottleneck_detector import (
    detect_reasoning_bottlenecks,
    find_isolated_clusters,
    trace_reasoning_path,
    multi_hop_vulnerability_analysis,
    generate_diagnostic_summary,
)
from .kg_rewirer import (
    suggest_new_connections,
    apply_suggestions,
    evaluate_rewiring_impact,
    auto_fix,
)
from .reasoning_evaluator import (
    generate_multi_hop_questions,
    evaluate_reachability,
    compare_before_after,
)
from .report_generator import (
    generate_text_report,
    generate_markdown_report,
)
from .visualization import (
    plot_kg_overview,
    plot_bottleneck_subgraph,
    plot_reasoning_path,
    plot_before_after_comparison,
    plot_relation_curvature_heatmap,
    plot_health_dashboard,
)

__all__ = [
    "load_from_triples",
    "load_from_json",
    "load_freebase_subset",
    "load_wn18rr_subset",
    "create_sample_kg",
    "get_kg_stats",
    "to_undirected",
    "compute_kg_curvature",
    "compute_relation_curvature_stats",
    "compute_entity_type_curvature",
    "identify_bridge_entities",
    "detect_reasoning_bottlenecks",
    "find_isolated_clusters",
    "trace_reasoning_path",
    "multi_hop_vulnerability_analysis",
    "generate_diagnostic_summary",
    "suggest_new_connections",
    "apply_suggestions",
    "evaluate_rewiring_impact",
    "auto_fix",
    "generate_multi_hop_questions",
    "evaluate_reachability",
    "compare_before_after",
    "generate_text_report",
    "generate_markdown_report",
    "plot_kg_overview",
    "plot_bottleneck_subgraph",
    "plot_reasoning_path",
    "plot_before_after_comparison",
    "plot_relation_curvature_heatmap",
    "plot_health_dashboard",
]
