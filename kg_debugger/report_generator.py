"""
Generate human-readable diagnostic reports (text and Markdown).
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _results_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    d = root / "results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_text_report(
    diagnostic_summary: Dict[str, Any],
    suggestions: Optional[pd.DataFrame] = None,
    reasoning_eval: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a formatted text report. Sections: Executive Summary, Health Score, Top Bottlenecks,
    Bridge Entities, Knowledge Islands, Multi-Hop Analysis, Recommended Fixes, Expected Impact.
    Writes to results/kg_diagnostic_report.txt.

    Args:
        diagnostic_summary: From generate_diagnostic_summary.
        suggestions: DataFrame from suggest_new_connections (optional).
        reasoning_eval: Dict with e.g. overall_reachability, comparison (optional).

    Returns:
        Path to the written file.
    """
    out = []
    out.append("=" * 60)
    out.append("KNOWLEDGE GRAPH DIAGNOSTIC REPORT")
    out.append("=" * 60)
    out.append("")
    out.append("--- Executive Summary ---")
    health = diagnostic_summary.get("overall_health_score", 0)
    out.append(f"Overall health score: {health}/100")
    out.append(f"Total bottleneck edges: {diagnostic_summary.get('total_bottlenecks', 0)}")
    out.append(f"Multi-hop vulnerability: {diagnostic_summary.get('multi_hop_vulnerability_pct', 0):.1f}%")
    out.append("")
    out.append("--- Top Bottlenecks ---")
    for i, e in enumerate(diagnostic_summary.get("worst_bottleneck_edges", [])[:10], 1):
        out.append(f"  {i}. {e.get('source_label', '')} --[{e.get('relation', '')}]--> {e.get('target_label', '')} (curvature={e.get('curvature', 0):.3f}, impact={e.get('impact_score', 0):.4f})")
    out.append("")
    out.append("--- Bridge Entities at Risk ---")
    for b in diagnostic_summary.get("bridge_entities", [])[:10]:
        out.append(f"  - {b.get('label', '')} ({b.get('type', '')}): curvature={b.get('curvature', 0):.3f}, betweenness={b.get('betweenness', 0):.4f}")
    out.append("")
    out.append("--- Knowledge Islands ---")
    for c in diagnostic_summary.get("isolated_clusters", []):
        out.append(f"  Cluster {c.get('cluster_id', 0)}: {c.get('size', 0)} nodes, {len(c.get('bottleneck_edges_to_other', []))} bottleneck edges to rest")
    out.append("")
    out.append("--- Multi-Hop Reasoning Analysis ---")
    out.append(f"  Sampled paths with bottleneck: {diagnostic_summary.get('multi_hop_vulnerability_pct', 0):.1f}%")
    if reasoning_eval:
        out.append(f"  Overall reachability score: {reasoning_eval.get('overall_reachability', 0):.3f}")
    out.append("")
    out.append("--- Recommended Fixes ---")
    if suggestions is not None and not suggestions.empty:
        for i, row in suggestions.head(15).iterrows():
            out.append(f"  - Add edge: {row.get('source_label', '')} --[{row.get('suggested_relation', '')}]--> {row.get('target_label', '')} (priority={row.get('priority', '')})")
    else:
        out.append("  (No suggestions generated)")
    out.append("")
    out.append("--- Expected Impact ---")
    if reasoning_eval and "improvement" in reasoning_eval:
        out.append(f"  Expected improvement after rewiring: {reasoning_eval.get('improvement', 0):.2f}")
    out.append("")
    out.append("=" * 60)

    path = _results_dir() / "kg_diagnostic_report.txt"
    path.write_text("\n".join(out), encoding="utf-8")
    logger.info("Wrote text report to %s", path)
    return str(path)


def generate_markdown_report(
    diagnostic_summary: Dict[str, Any],
    suggestions: Optional[pd.DataFrame] = None,
    reasoning_eval: Optional[Dict[str, Any]] = None,
    figures_dir: Optional[str] = None,
) -> str:
    """
    Same content as Markdown with optional embedded figure references.
    Writes to results/kg_diagnostic_report.md.

    Args:
        diagnostic_summary: From generate_diagnostic_summary.
        suggestions: Optional suggestions DataFrame.
        reasoning_eval: Optional reasoning evaluation dict.
        figures_dir: Optional path to figures (e.g. "figures") for image links.

    Returns:
        Path to the written file.
    """
    fig = figures_dir or "figures"
    out = []
    out.append("# Knowledge Graph Diagnostic Report\n")
    out.append("## Executive Summary")
    out.append(f"- **Health score:** {diagnostic_summary.get('overall_health_score', 0)}/100")
    out.append(f"- **Bottleneck edges:** {diagnostic_summary.get('total_bottlenecks', 0)}")
    out.append(f"- **Multi-hop vulnerability:** {diagnostic_summary.get('multi_hop_vulnerability_pct', 0):.1f}%\n")
    out.append("## Top Bottlenecks")
    out.append("| # | Source | Relation | Target | Curvature | Impact |")
    out.append("|---|--------|----------|--------|------------|--------|")
    for i, e in enumerate(diagnostic_summary.get("worst_bottleneck_edges", [])[:10], 1):
        out.append(f"| {i} | {e.get('source_label', '')} | {e.get('relation', '')} | {e.get('target_label', '')} | {e.get('curvature', 0):.3f} | {e.get('impact_score', 0):.4f} |")
    out.append("\n## Bridge Entities")
    for b in diagnostic_summary.get("bridge_entities", [])[:10]:
        out.append(f"- **{b.get('label', '')}** ({b.get('type', '')}): curvature={b.get('curvature', 0):.3f}")
    out.append("\n## Knowledge Islands")
    for c in diagnostic_summary.get("isolated_clusters", []):
        out.append(f"- Cluster {c.get('cluster_id', 0)}: {c.get('size', 0)} nodes")
    out.append("\n## Recommended Fixes")
    if suggestions is not None and not suggestions.empty:
        out.append("| Source | Target | Relation | Priority |")
        out.append("|--------|--------|----------|----------|")
        for _, row in suggestions.head(15).iterrows():
            out.append(f"| {row.get('source_label', '')} | {row.get('target_label', '')} | {row.get('suggested_relation', '')} | {row.get('priority', '')} |")
    out.append("\n## Figures")
    out.append(f"- Overview: `{fig}/kg_overview.png`")
    out.append(f"- Health dashboard: `{fig}/kg_health_dashboard.png`")

    path = _results_dir() / "kg_diagnostic_report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    logger.info("Wrote Markdown report to %s", path)
    return str(path)
