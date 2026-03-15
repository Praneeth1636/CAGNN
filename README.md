<div align="center">

# 🔬 CAGNN — Curvature-Aware Graph Neural Networks

**Detect and fix reasoning bottlenecks in graphs using differential geometry**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.4%2B-3C2179.svg)](https://pytorch-geometric.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#overview) · [Knowledge Graph Debugger](#-knowledge-graph-debugger) · [Quick Start](#-quick-start) · [Results](#-results) · [Architecture](#-architecture) · [References](#-references)

</div>

---

## Overview

Graph Neural Networks (GNNs) suffer from **over-squashing** — when information from distant nodes must pass through narrow bottleneck edges, signals get compressed and distorted. This limits GNNs' ability to perform long-range reasoning.

**CAGNN** tackles this from a geometric perspective. Using **Ollivier-Ricci curvature** and **local Gromov δ-hyperbolicity**, we identify exactly where bottlenecks occur, prove they correlate with GNN misclassifications, and fix them through curvature-guided graph rewiring.

This project has two components:

| Component | What it does | Who it's for |
|-----------|-------------|--------------|
| **GNN Analysis Pipeline** | Trains GCN/GAT models, computes curvature, detects over-squashing, rewires graphs | ML researchers studying GNN limitations |
| **Knowledge Graph Debugger** | Finds weak connections in any knowledge graph that block multi-hop reasoning, suggests fixes | Knowledge engineers, anyone building KGs |

---

## The Problem

<div align="center">

```
Node A ──── Node B ──── Node C ──── Node D
  │                                    │
  ├── 50 nodes                   40 nodes ──┤
  │   in this                    in this    │
  │   cluster                    cluster    │
  │                                         │
  └─────────────────────────────────────────┘
        All information must squeeze through
        the B─C edge → OVER-SQUASHING
```

</div>

In a GNN, each node gathers information from its neighbors. After *k* layers, a node has heard from everything within *k* hops. But if the graph has **bottleneck edges** (narrow bridges between large subgraphs), information from many nodes gets crushed through a single edge. The result: the GNN can't reason about distant relationships.

**Our approach:** Compute Ollivier-Ricci curvature on every edge. Edges with **negative curvature** are bottlenecks. We prove these correlate with misclassifications, then add shortcut edges to open up the bottlenecks.

---

## 🔍 Knowledge Graph Debugger

The practical application of our curvature analysis. Upload any knowledge graph and get a full diagnostic report.

### What it does

1. **Loads your knowledge graph** — supports CSV triples, JSON, or standard benchmarks (FB15k-237, WN18RR)
2. **Computes curvature on every edge** — identifies which connections are structurally weak
3. **Detects reasoning bottlenecks** — finds edges where multi-hop queries will fail
4. **Identifies bridge entities** — single points of failure connecting knowledge clusters
5. **Finds knowledge islands** — poorly connected subgraphs that block information flow
6. **Generates multi-hop questions** — automatically tests if the KG supports complex reasoning
7. **Suggests specific fixes** — recommends new connections with priority rankings
8. **Auto-fixes and evaluates** — applies rewiring and shows before/after improvement

### Use cases

- **Company wikis**: Find where knowledge silos exist between departments
- **Ontologies**: Detect weak taxonomic links that break inheritance reasoning
- **Product catalogs**: Find disconnected product relationships that hurt recommendation systems
- **Research knowledge bases**: Identify missing cross-domain connections

### Run the Streamlit app

```bash
streamlit run app.py
```

### Run the demo

```bash
python demo_kg.py
```

This runs the full pipeline on a built-in sample knowledge graph (a fictional tech company with ~80 entities across teams, projects, technologies, and locations) and generates a complete diagnostic report.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ ([installation guide](https://pytorch.org/get-started/locally/))
- CUDA-capable GPU (optional, but recommended)

### Installation

```bash
git clone https://github.com/Praneeth1636/CAGNN.git
cd CAGNN
pip install -r requirements.txt
```

> **Note:** PyTorch Geometric may need a specific installation for your CUDA version. See the [PyG install guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Run the GNN analysis pipeline

```bash
# Quick test (50 epochs, 1 run)
python main.py --quick

# Full run on Cora
python main.py --dataset Cora --epochs 200 --num_runs 3

# Run on CiteSeer or PubMed
python main.py --dataset CiteSeer --epochs 200 --num_runs 3
```

### Run the Knowledge Graph Debugger

```bash
# Demo with built-in sample KG
python demo_kg.py

# Interactive web app
streamlit run app.py
```

### Explore interactively

```bash
jupyter notebook notebooks/exploration.ipynb
```

---

## 📊 Results

### GNN Over-Squashing Analysis (Cora Dataset)

| Model | Original Accuracy | Curvature Rewired | Random Rewired |
|-------|:-:|:-:|:-:|
| GCN | 81.5% | **83.2%** (+1.7%) | 81.3% (-0.2%) |
| GAT | 83.1% | **84.4%** (+1.3%) | 82.9% (-0.2%) |

> Curvature-guided rewiring consistently outperforms both the original graph and random rewiring, confirming that geometry identifies the right edges to modify.

### Key findings

**1. Curvature predicts GNN failures.** Nodes in negatively curved (bottleneck) regions are misclassified at significantly higher rates than nodes in positively curved regions. The Pearson correlation between node curvature and classification accuracy is statistically significant (p < 0.05).

**2. Curvature-based rewiring works; random rewiring doesn't.** Adding the same number of random edges produces no improvement — confirming that it's the *geometric targeting* that matters, not just adding more connections.

**3. Local hyperbolicity captures bottleneck severity.** Gromov δ-hyperbolicity computed on 2-hop neighborhoods correlates with GNN performance degradation, validating the use of local (scalable) geometric measures over global (expensive) ones.

### Knowledge Graph Debugger Results (Sample KG)

| Metric | Before Fix | After Fix |
|--------|:-:|:-:|
| Bottleneck edges | 23 | 8 |
| Multi-hop reachability | 64% | 87% |
| Bridge entities (single points of failure) | 7 | 2 |
| Health score | 58/100 | 84/100 |

---

## 🏗 Architecture

```
CAGNN/
├── main.py                    # GNN analysis pipeline entry point
├── demo_kg.py                 # KG Debugger demo script
├── app.py                     # Streamlit web interface
├── requirements.txt
│
├── src/                       # Core GNN analysis modules
│   ├── models.py              #   GCN and GAT implementations
│   ├── curvature.py           #   Ollivier-Ricci + Gromov hyperbolicity
│   ├── rewiring.py            #   Curvature-based graph rewiring
│   ├── analysis.py            #   Over-squashing detection + correlation
│   ├── visualization.py       #   Publication-quality plots
│   └── utils.py               #   Data loading, config, helpers
│
├── kg_debugger/               # Knowledge Graph Debugger
│   ├── kg_loader.py           #   Load KGs from CSV, JSON, benchmarks
│   ├── kg_curvature.py        #   KG-specific curvature analysis
│   ├── bottleneck_detector.py #   Reasoning bottleneck detection
│   ├── kg_rewirer.py          #   Suggest and apply fixes
│   ├── reasoning_evaluator.py #   Multi-hop reasoning evaluation
│   ├── report_generator.py    #   Diagnostic report generation
│   └── visualization.py       #   KG-specific visualizations
│
├── notebooks/
│   └── exploration.ipynb      # Interactive walkthrough
├── figures/                   # Generated plots
├── results/                   # CSV results and reports
└── sample_data/               # Sample knowledge graphs
```

---

## Methods

### Ollivier-Ricci Curvature

For each edge (u, v), we compute:

```
κ(u,v) = 1 - W₁(mᵤ, mᵥ) / d(u,v)
```

Where `mᵤ, mᵥ` are probability distributions over the neighborhoods of u and v, and `W₁` is the Wasserstein-1 (Earth Mover's) distance. Intuitively:

- **κ > 0 (positive):** Nodes share many neighbors → tight community, information flows easily
- **κ ≈ 0 (flat):** Grid-like structure
- **κ < 0 (negative):** Neighborhoods are disjoint → bottleneck edge, information gets compressed

### Local Gromov δ-Hyperbolicity

For each node, we compute hyperbolicity on its k-hop neighborhood by sampling quadruples (x, y, z, w) and measuring:

```
δ = (max_sum - second_max_sum) / 2
```

where the three sums are `d(x,y)+d(z,w)`, `d(x,z)+d(y,w)`, `d(x,w)+d(y,z)`. This is computed **locally** (per-node neighborhood) rather than globally, reducing complexity from O(n⁴) to O(k⁴) where k is the neighborhood size.

### Curvature-Based Rewiring

1. Sort edges by curvature (most negative first)
2. Identify bottleneck nodes (incident to negatively curved edges)
3. Add shortcut edges between 2-hop neighbors of bottleneck nodes
4. Compare against random rewiring baseline (same number of edges)

---

## Supported Input Formats (KG Debugger)

| Format | Description | Example |
|--------|-------------|---------|
| **CSV/TSV triples** | `head, relation, tail` per row | `Einstein, born_in, Ulm` |
| **JSON** | `{"entities": [...], "relations": [...]}` | Structured with types |
| **FB15k-237** | Standard Freebase benchmark (auto-download) | Built-in via PyG |
| **WN18RR** | Standard WordNet benchmark (auto-download) | Built-in via PyG |
| **Sample KG** | Built-in tech company KG for testing | `create_sample_kg()` |

---

## CLI Reference

### `main.py` — GNN Analysis

```
python main.py [OPTIONS]

Options:
  --dataset     Dataset name: Cora, CiteSeer, PubMed     [default: Cora]
  --epochs      Training epochs                           [default: 200]
  --num_runs    Runs for rewiring comparison              [default: 3]
  --device      cuda or cpu                               [default: auto]
  --quick       Quick test mode (50 epochs, 1 run)
```

### `demo_kg.py` — KG Debugger Demo

```
python demo_kg.py
```

Runs the full debugger pipeline on the sample KG, generates reports in `results/` and figures in `figures/`.

### `app.py` — Streamlit Dashboard

```
streamlit run app.py
```

Interactive web interface with upload, analysis, visualization, and fix suggestions.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Deep Learning** | PyTorch, PyTorch Geometric |
| **Graph Analysis** | NetworkX, GraphRicciCurvature |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Interface** | Streamlit |
| **Data** | Pandas, NumPy, SciPy |
| **Benchmarks** | Cora, CiteSeer, PubMed, FB15k-237, WN18RR |

---

## References

1. **Topping, J., Di Giovanni, F., Chamberlain, B.P., Dong, X., & Bronstein, M.M. (2022).** "Understanding Over-Squashing and Bottlenecks on Graphs via Curvature." *ICLR 2022.*

2. **Alon, U. & Yahav, E. (2021).** "On the Bottleneck of Graph Neural Networks and its Practical Implications." *ICLR 2021.*

3. **Ollivier, Y. (2009).** "Ricci curvature of Markov chains on metric spaces." *Journal of Functional Analysis, 256(3), 810-864.*

4. **Ni, C.-C., Lin, Y.-Y., Luo, F., & Gao, J. (2019).** "Community Detection on Networks with Ricci Flow." *Scientific Reports.*

5. **Gromov, M. (1987).** "Hyperbolic Groups." *Essays in Group Theory, MSRI Publications.*

---

## Contributing

Contributions are welcome. If you find a bug or want to add a feature:

1. Fork the repo
2. Create a branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push and open a PR

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Praneeth Yashovardhan Kadem](https://github.com/Praneeth1636)

*Research project for geometric deep learning — NYU Computer Science*

</div>
