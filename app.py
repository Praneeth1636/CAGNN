"""
Streamlit web interface for the Knowledge Graph Debugger.
Run with: streamlit run app.py
"""

import io
import logging
from pathlib import Path

import pandas as pd
import streamlit as st

# Project root
ROOT = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(ROOT))

from src.curvature import get_node_curvature

from kg_debugger import (
    create_sample_kg,
    load_from_triples,
    load_from_json,
    load_freebase_subset,
    load_wn18rr_subset,
    get_kg_stats,
    compute_kg_curvature,
    compute_relation_curvature_stats,
    identify_bridge_entities,
    detect_reasoning_bottlenecks,
    find_isolated_clusters,
    trace_reasoning_path,
    multi_hop_vulnerability_analysis,
    generate_diagnostic_summary,
    suggest_new_connections,
    apply_suggestions,
    evaluate_rewiring_impact,
    auto_fix,
    generate_multi_hop_questions,
    evaluate_reachability,
    compare_before_after,
    generate_text_report,
    generate_markdown_report,
    plot_kg_overview,
    plot_bottleneck_subgraph,
    plot_reasoning_path,
    plot_before_after_comparison,
    plot_relation_curvature_heatmap,
    plot_health_dashboard,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _node_curvature_from_edges(G, G_curved, edge_curvatures):
    """Build full node curvature for all nodes in G from G_curved and edge_curvatures."""
    nc = get_node_curvature(G_curved) if G_curved.number_of_nodes() else {}
    return {n: nc.get(n, 0.0) for n in G.nodes()}


@st.cache_data(ttl=3600)
def _load_kg(source: str, file_content=None, file_name=None, num_entities: int = 500):
    """Load KG by source; return (G or None, error message)."""
    try:
        if source == "Sample (tech company)":
            return create_sample_kg(), None
        if source == "FB15k-237 subset":
            return load_freebase_subset(num_entities=num_entities), None
        if source == "WN18RR subset":
            return load_wn18rr_subset(num_entities=num_entities), None
        if source == "Upload CSV/TSV triples" and file_content:
            buf = io.BytesIO(file_content)
            return load_from_triples(buf), None
        if source == "Upload JSON" and file_content:
            buf = io.BytesIO(file_content)
            return load_from_json(buf), None
    except Exception as e:
        return None, str(e)
    return None, "Choose a source and/or upload a file."


@st.cache_data(ttl=3600)
def _compute_curvature_cached(graph_hash: str, alpha: float):
    """Compute curvature; cache by graph_hash and alpha. Caller must pass current G via session_state."""
    G = st.session_state.get("G")
    if G is None or G.number_of_nodes() == 0:
        return None, None, None, None
    try:
        G_curved, edge_curvatures = compute_kg_curvature(G, alpha=alpha)
        node_curvature = _node_curvature_from_edges(G, G_curved, edge_curvatures)
        rel_stats = compute_relation_curvature_stats(G_curved, edge_curvatures)
        return G_curved, edge_curvatures, node_curvature, rel_stats
    except Exception as e:
        st.error(f"Curvature computation failed: {e}")
        return None, None, None, None


def _graph_hash(G):
    import hashlib
    import networkx as nx
    n, m = G.number_of_nodes(), G.number_of_edges()
    edges = list(G.edges())[:100]
    data = f"{n}_{m}_{edges}"
    return hashlib.md5(data.encode()).hexdigest()


def main():
    st.set_page_config(page_title="KG Debugger", layout="wide")
    st.title("Knowledge Graph Debugger")
    st.markdown("Find and fix weak connections in knowledge graphs using curvature analysis.")

    # Sidebar
    with st.sidebar:
        st.header("Load knowledge graph")
        source = st.selectbox(
            "Source",
            ["Sample (tech company)", "FB15k-237 subset", "WN18RR subset", "Upload CSV/TSV triples", "Upload JSON"],
            index=0,
        )
        file_upload = None
        if "Upload" in source:
            file_upload = st.file_uploader("Upload file", type=["csv", "tsv", "json"])
        num_entities = 500
        if "subset" in source.lower():
            num_entities = st.number_input("Subset size", min_value=100, max_value=2000, value=500, step=100)

        if st.button("Load KG"):
            content = file_upload.read() if file_upload else None
            name = file_upload.name if file_upload else None
            G, err = _load_kg(source, content, name, num_entities)
            if err:
                st.error(err)
            else:
                st.session_state["G"] = G
                st.session_state["G_curved"] = None
                st.success(f"Loaded {G.number_of_nodes()} entities, {G.number_of_edges()} relations.")

        st.header("Parameters")
        alpha = st.slider("Curvature alpha", 0.1, 0.9, 0.5, 0.1)
        bottleneck_threshold = st.slider("Bottleneck threshold", -0.5, 0.0, -0.1, 0.05)
        rewiring_budget = st.number_input("Rewiring budget", min_value=1, max_value=50, value=20)

    G = st.session_state.get("G")
    if G is None:
        st.info("Load a knowledge graph from the sidebar to start.")
        with st.expander("What is Ollivier-Ricci curvature?"):
            st.markdown("""
            Ollivier-Ricci curvature measures how much the neighborhood of two connected nodes
            overlaps. Negative curvature indicates bottleneck regions where information is
            squeezed and multi-hop reasoning can fail.
            """)
        return

    # Curvature (cached by graph hash + alpha)
    g_hash = _graph_hash(G)
    with st.spinner("Computing curvature…"):
        G_curved, edge_curvatures, node_curvature, relation_stats = _compute_curvature_cached(g_hash, alpha)
    if G_curved is None or edge_curvatures is None:
        st.warning("Could not compute curvature. Try the sample KG or check dependencies.")
        return

    # Diagnostic summary (recompute when curvature changes)
    diagnostic = generate_diagnostic_summary(G, edge_curvatures, node_curvature)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Bottleneck Analysis",
        "Multi-Hop Reasoning",
        "Fix Suggestions",
        "Report",
    ])

    with tab1:
        st.subheader("KG statistics")
        stats_text = io.StringIO()
        get_kg_stats(G, stream=stats_text)
        st.text(stats_text.getvalue())

        st.subheader("Health score")
        health = diagnostic.get("overall_health_score", 0)
        st.metric("Overall health", f"{health}/100", delta=None)
        st.progress(health / 100.0)

        st.subheader("Graph overview")
        with st.spinner("Plotting…"):
            plot_kg_overview(G, node_curvature, "KG Overview", edge_curvatures=edge_curvatures)
        st.image(str(FIG_DIR / "kg_overview.png"), use_container_width=True)

    with tab2:
        top_k = 20
        bottlenecks_df = detect_reasoning_bottlenecks(G, edge_curvatures, node_curvature, top_k=top_k)
        st.subheader("Top bottleneck edges")
        if not bottlenecks_df.empty:
            st.dataframe(bottlenecks_df, use_container_width=True)
            selected = st.selectbox(
                "Select a bottleneck to visualize",
                range(len(bottlenecks_df)),
                format_func=lambda i: f"{bottlenecks_df.iloc[i]['source_label']} --[{bottlenecks_df.iloc[i]['relation']}]--> {bottlenecks_df.iloc[i]['target_label']}",
            )
            u = int(bottlenecks_df.iloc[selected]["source"])
            v = int(bottlenecks_df.iloc[selected]["target"])
            plot_bottleneck_subgraph(G, [(u, v)], edge_curvatures, "Bottleneck subgraph")
            st.image(str(FIG_DIR / "kg_bottleneck_subgraph.png"), use_container_width=True)
        else:
            st.info("No bottlenecks detected.")

        st.subheader("Bridge entities")
        bridge_entities = identify_bridge_entities(G, node_curvature, threshold=bottleneck_threshold)
        if bridge_entities:
            bridge_data = [{"Label": x[3], "Type": x[4], "Curvature": f"{x[1]:.4f}", "Betweenness": f"{x[2]:.4f}"} for x in bridge_entities[:20]]
            st.dataframe(pd.DataFrame(bridge_data), use_container_width=True)
        else:
            st.info("No bridge entities in this range.")

        st.subheader("Relation curvature")
        if relation_stats is not None and not relation_stats.empty:
            plot_relation_curvature_heatmap(relation_stats)
            st.image(str(FIG_DIR / "kg_relation_heatmap.png"), use_container_width=True)

    with tab3:
        num_q = 20
        questions = generate_multi_hop_questions(G, num_questions=num_q, min_hops=2, max_hops=4)
        overall, per_scores = evaluate_reachability(G, questions, edge_curvatures)
        st.session_state["overall_reachability"] = overall
        st.session_state["questions"] = questions
        st.metric("Overall reachability score", f"{overall:.3f}", delta=None)
        st.caption("Based on curvature-weighted signal propagation.")

        st.subheader("Sample questions and paths")
        for i, q in enumerate(questions[:10]):
            with st.expander(f"Q{i+1}: {q['question_text'][:80]}…"):
                path_data = trace_reasoning_path(G, q["start"], q["end"], edge_curvatures)
                if path_data:
                    st.write("Path:", " → ".join([str(p[0]) for p in path_data]))
                    st.write("Reachability score:", f"{per_scores[i]:.3f}" if i < len(per_scores) else "—")
                else:
                    st.write("No path.")

        vuln_stats, _ = multi_hop_vulnerability_analysis(G, edge_curvatures, num_samples=min(200, G.number_of_nodes() * 2))
        st.subheader("Multi-hop vulnerability")
        st.write(f"Share of sampled long paths with a bottleneck: **{vuln_stats.get('pct_paths_with_bottleneck', 0):.1f}%**")

    with tab4:
        suggestions_df = suggest_new_connections(G, edge_curvatures, node_curvature, max_suggestions=rewiring_budget)
        st.subheader("Suggested new connections")
        if not suggestions_df.empty:
            st.dataframe(suggestions_df, use_container_width=True)
            if st.button("Apply all suggestions"):
                G_rewired = apply_suggestions(G, suggestions_df)
                st.session_state["G_rewired"] = G_rewired
                report = evaluate_rewiring_impact(G, G_rewired, edge_curvatures)
                st.session_state["rewiring_report"] = report
                st.success("Rewiring applied. See comparison below.")
        else:
            st.info("No suggestions generated.")

        if st.session_state.get("G_rewired") is not None:
            G_rewired = st.session_state["G_rewired"]
            report = st.session_state.get("rewiring_report", {})
            st.subheader("Before / After")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Health before", report.get("health_before", 0), delta=None)
            with c2:
                st.metric("Health after", report.get("health_after", 0), delta=report.get("improvement"))
            with c3:
                st.metric("Bottlenecks before → after", f"{report.get('before_bottlenecks', 0)} → {report.get('after_bottlenecks', 0)}", delta=None)

            with st.spinner("Computing curvature for rewired graph…"):
                try:
                    G_curved_r, edge_curv_r = compute_kg_curvature(G_rewired, alpha=alpha)
                    node_curv_r = _node_curvature_from_edges(G_rewired, G_curved_r, edge_curv_r)
                    plot_before_after_comparison(G, G_rewired, node_curvature, node_curv_r)
                    st.image(str(FIG_DIR / "kg_before_after.png"), use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot comparison: {e}")

            buf = io.StringIO()
            buf.write("head,relation,tail\n")
            for u, v, d in G_rewired.edges(data=True):
                buf.write(f"{u},{d.get('relation','related_to')},{v}\n")
            st.download_button("Download fixed KG (CSV triples)", buf.getvalue(), file_name="kg_fixed.csv", mime="text/csv")

    with tab5:
        suggestions_for_report = suggest_new_connections(G, edge_curvatures, node_curvature, max_suggestions=rewiring_budget)
        reasoning_eval = {"overall_reachability": st.session_state.get("overall_reachability", 0.0)}
        if st.session_state.get("G_rewired") is not None and st.session_state.get("rewiring_report"):
            reasoning_eval["improvement"] = st.session_state["rewiring_report"].get("improvement", 0)
        if st.button("Generate full report"):
            with st.spinner("Generating report…"):
                plot_health_dashboard(diagnostic)
                generate_text_report(diagnostic, suggestions_for_report, reasoning_eval)
                generate_markdown_report(diagnostic, suggestions_for_report, reasoning_eval, str(FIG_DIR))
            st.success("Report saved to results/kg_diagnostic_report.txt and .md")
        try:
            md_path = RESULTS_DIR / "kg_diagnostic_report.md"
            if md_path.exists():
                st.markdown(md_path.read_text(encoding="utf-8"))
                st.download_button("Download report (Markdown)", md_path.read_text(encoding="utf-8"), file_name="kg_diagnostic_report.md", mime="text/markdown")
            else:
                st.info("Click 'Generate full report' to create the report.")
        except Exception as e:
            st.info("Generate the report first.")

    with st.sidebar:
        with st.expander("What is Ollivier-Ricci curvature?"):
            st.markdown("Ollivier-Ricci curvature measures bottleneck regions in graphs. Negative values indicate weak connections that hurt multi-hop reasoning.")


if __name__ == "__main__":
    main()
