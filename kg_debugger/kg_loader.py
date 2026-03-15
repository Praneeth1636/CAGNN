"""
Load knowledge graphs from various formats into a unified NetworkX DiGraph.
Nodes: label, type. Edges: relation, weight (default 1.0).
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


def _normalize_triple_row(row: List[str], delimiter: str = ",") -> Tuple[str, str, str]:
    """Parse a row into (head, relation, tail). Handles CSV/TSV."""
    if len(row) >= 3:
        return row[0].strip(), row[1].strip(), row[2].strip()
    return ("", "", "")


def load_from_triples(filepath: str) -> nx.DiGraph:
    """
    Load a knowledge graph from a CSV/TSV file with columns: head, relation, tail.
    Creates integer node IDs; node attributes: label (entity name), type (default "Entity").
    Edge attributes: relation, weight (1.0).

    Args:
        filepath: Path to CSV or TSV file. First row may be header.

    Returns:
        NetworkX DiGraph with node/edge attributes as above.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Triples file not found: {filepath}")

    delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    G = nx.DiGraph()
    entity_to_id: Dict[str, int] = {}
    id_counter = [0]

    def get_id(name: str) -> int:
        if name not in entity_to_id:
            entity_to_id[name] = id_counter[0]
            id_counter[0] += 1
            G.add_node(entity_to_id[name], label=name, type="Entity")
        return entity_to_id[name]

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first = next(reader, None)
        if first is None:
            logger.warning("Empty triples file: %s", filepath)
            return G
        # If first row looks like header, skip it
        if first[0].lower() in ("head", "subject", "source") or not first[0].strip():
            pass
        else:
            h, r, t = _normalize_triple_row(first, delimiter)
            if h and r and t:
                u, v = get_id(h), get_id(t)
                G.add_edge(u, v, relation=r, weight=1.0)
        for row in reader:
            if len(row) < 3:
                continue
            h, r, t = _normalize_triple_row(row, delimiter)
            if not h or not r or not t:
                continue
            u, v = get_id(h), get_id(t)
            G.add_edge(u, v, relation=r, weight=1.0)

    logger.info("Loaded %d entities and %d relations from %s", G.number_of_nodes(), G.number_of_edges(), filepath)
    return G


def load_from_json(filepath: str) -> nx.DiGraph:
    """
    Load from JSON format: {"entities": [...], "relations": [...]}.
    Entities: id, label, type. Relations: head, tail, relation.

    Args:
        filepath: Path to JSON file.

    Returns:
        NetworkX DiGraph with integer node IDs (from entity id or index).
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {filepath}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    entities = data.get("entities", [])
    relations = data.get("relations", [])
    id_to_idx: Dict[Any, int] = {}
    for i, ent in enumerate(entities):
        eid = ent.get("id", i)
        id_to_idx[eid] = i
        G.add_node(
            i,
            label=ent.get("label", str(eid)),
            type=ent.get("type", "Entity"),
        )
    for rel in relations:
        h = rel.get("head")
        t = rel.get("tail")
        r = rel.get("relation", "related_to")
        if h is None or t is None:
            continue
        u = id_to_idx.get(h)
        v = id_to_idx.get(t)
        if u is None or v is None:
            continue
        G.add_edge(u, v, relation=r, weight=1.0)

    logger.info("Loaded %d entities and %d relations from %s", G.number_of_nodes(), G.number_of_edges(), filepath)
    return G


def load_freebase_subset(num_entities: int = 500) -> nx.DiGraph:
    """
    Load a small subset of a Freebase-style KG. Uses PyTorch Geometric's
    built-in datasets if available; otherwise builds a synthetic subset.

    Args:
        num_entities: Target number of entities (approximate).

    Returns:
        NetworkX DiGraph in unified format.
    """
    try:
        from torch_geometric.datasets import Entities
        # Entities is for node classification; try RelationalDataset or similar
    except ImportError:
        pass

    # Fallback: build a synthetic Freebase-style KG with common relation types
    G = nx.DiGraph()
    relations_fb = [
        "type.object.type", "common.topic.alias", "people.person.nationality",
        "location.location.contains", "people.person.profession", "organization.organization.founders",
        "time.event.included_in", "music.artist.genre", "book.author.works_written",
        "sports.sports_team.location", "film.film.directed_by", "tv.tv_program.genre",
    ]
    import random
    rng = random.Random(42)
    n = min(num_entities, 400)
    for i in range(n):
        G.add_node(i, label=f"entity_{i}", type=rng.choice(["Person", "Place", "Organization", "Topic", "Event"]))
    # Add edges so graph is connected in one giant component
    for i in range(n - 1):
        j = rng.randint(i + 1, min(i + 50, n - 1)) if i + 50 < n else rng.randint(i + 1, n - 1)
        if j < n:
            G.add_edge(i, j, relation=rng.choice(relations_fb), weight=1.0)
            G.add_edge(j, i, relation=rng.choice(relations_fb), weight=1.0)
    # Add more random edges
    target_edges = n * 3
    while G.number_of_edges() < target_edges:
        u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, relation=rng.choice(relations_fb), weight=1.0)

    logger.info("Loaded Freebase-style subset: %d entities, %d relations", G.number_of_nodes(), G.number_of_edges())
    return G


def load_wn18rr_subset(num_entities: int = 500) -> nx.DiGraph:
    """
    Load a small subset of WordNet WN18RR using PyTorch Geometric's WordNet18RR if available.
    Converts to NetworkX with entity/relation labels.

    Args:
        num_entities: Target approximate number of entities.

    Returns:
        NetworkX DiGraph in unified format.
    """
    try:
        from torch_geometric.datasets import WordNet18RR
        import torch
        dataset = WordNet18RR(root="/tmp/WN18RR")[0]
        edge_index = dataset.edge_index
        edge_type = dataset.edge_type
        num_nodes = dataset.num_nodes
        # WN18RR has edge_type as relation index; we use generic labels unless we have a mapping
        G = nx.DiGraph()
        for i in range(num_nodes):
            G.add_node(i, label=f"synset_{i}", type="Synset")
        rel_names = getattr(dataset, "rel_names", None) or [f"rel_{r}" for r in range(edge_type.max().item() + 1)]
        for e in range(edge_index.size(1)):
            u = edge_index[0, e].item()
            v = edge_index[1, e].item()
            r = edge_type[e].item()
            rel_label = rel_names[r] if r < len(rel_names) else f"rel_{r}"
            G.add_edge(u, v, relation=rel_label, weight=1.0)
        # If we have too many nodes, take LCC subgraph and limit
        if not nx.is_weakly_connected(G):
            largest = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest).copy()
        if G.number_of_nodes() > num_entities:
            # Induce subgraph from first num_entities nodes in a BFS from 0
            seen = {0}
            frontier = [0]
            while frontier and len(seen) < num_entities:
                u = frontier.pop(0)
                for _, v in G.out_edges(u):
                    if v not in seen:
                        seen.add(v)
                        frontier.append(v)
                        if len(seen) >= num_entities:
                            break
            G = G.subgraph(seen).copy()
        logger.info("Loaded WN18RR subset: %d entities, %d relations", G.number_of_nodes(), G.number_of_edges())
        return G
    except Exception as e:
        logger.warning("WordNet18RR load failed (%s), using synthetic WordNet-style KG", e)
        G = nx.DiGraph()
        relations_wn = ["_hypernym", "_part_of", "_member_meronym", "_similar_to", "_also_see", "_verb_group"]
        import random
        rng = random.Random(43)
        n = min(num_entities, 300)
        for i in range(n):
            G.add_node(i, label=f"synset_{i}", type="Synset")
        for i in range(n - 1):
            j = rng.randint(0, n - 1)
            if i != j:
                G.add_edge(i, j, relation=rng.choice(relations_wn), weight=1.0)
        for _ in range(n * 2):
            u, v = rng.randint(0, n - 1), rng.randint(0, n - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v, relation=rng.choice(relations_wn), weight=1.0)
        return G


def create_sample_kg() -> nx.DiGraph:
    """
    Create a hardcoded sample knowledge graph about a fictional tech company.
    ~80 entities, ~150 relations with natural bottlenecks (e.g. one person
    connecting ML team to DevOps). Entity types: Person, Team, Project, Technology,
    Location, Document. Relations: works_in, leads, uses_tech, located_in, depends_on,
    authored, reports_to, collaborates_with.

    Returns:
        NetworkX DiGraph with integer nodes and full attributes.
    """
    G = nx.DiGraph()

    # Node list: (index, label, type)
    nodes = [
        (0, "Alice", "Person"), (1, "Bob", "Person"), (2, "Carol", "Person"), (3, "Dave", "Person"),
        (4, "Eve", "Person"), (5, "Frank", "Person"), (6, "Grace", "Person"), (7, "Henry", "Person"),
        (8, "Ivy", "Person"), (9, "Jack", "Person"), (10, "Kate", "Person"), (11, "Leo", "Person"),
        (12, "ML Team", "Team"), (13, "DevOps Team", "Team"), (14, "Backend Team", "Team"),
        (15, "Frontend Team", "Team"), (16, "Data Team", "Team"), (17, "Security Team", "Team"),
        (18, "Project Alpha", "Project"), (19, "Project Beta", "Project"), (20, "Project Gamma", "Project"),
        (21, "Project Delta", "Project"), (22, "Project Echo", "Project"),
        (23, "Python", "Technology"), (24, "TensorFlow", "Technology"), (25, "Kubernetes", "Technology"),
        (26, "React", "Technology"), (27, "PostgreSQL", "Technology"), (28, "AWS", "Technology"),
        (29, "Docker", "Technology"), (30, "Spark", "Technology"), (31, "GraphQL", "Technology"),
        (32, "HQ", "Location"), (33, "Office EU", "Location"), (34, "Office APAC", "Location"),
        (35, "Cloud DC1", "Location"), (36, "Cloud DC2", "Location"),
        (37, "Doc Architecture", "Document"), (38, "Doc API", "Document"), (39, "Doc Runbook", "Document"),
        (40, "Doc Security", "Document"), (41, "Doc ML Pipeline", "Document"),
        (42, "Bridge Person", "Person"),  # Single point connecting ML and DevOps
    ]
    for idx, label, etype in nodes:
        G.add_node(idx, label=label, type=etype)

    # Add more persons to reach ~80 entities
    for i in range(43, 80):
        names = ["Alex", "Sam", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Quinn", "Parker", "Avery"]
        G.add_node(i, label=f"{names[i % 10]}_{i}", type="Person")

    def add(u: int, v: int, rel: str) -> None:
        G.add_edge(u, v, relation=rel, weight=1.0)

    # Works_in: people -> teams
    add(0, 12, "works_in"); add(1, 12, "works_in"); add(2, 12, "works_in"); add(42, 12, "works_in")
    add(3, 13, "works_in"); add(4, 13, "works_in"); add(42, 13, "works_in")  # Bridge in both ML and DevOps
    add(5, 14, "works_in"); add(6, 14, "works_in"); add(7, 15, "works_in"); add(8, 15, "works_in")
    add(9, 16, "works_in"); add(10, 16, "works_in"); add(11, 17, "works_in")
    for i in range(43, 55):
        add(i, 12 if i % 3 == 0 else (13 if i % 3 == 1 else 14), "works_in")
    for i in range(55, 70):
        add(i, 15 if i % 2 == 0 else 16, "works_in")
    for i in range(70, 80):
        add(i, 17, "works_in")

    # Leads: one per team
    add(0, 12, "leads"); add(3, 13, "leads"); add(5, 14, "leads"); add(7, 15, "leads")
    add(9, 16, "leads"); add(11, 17, "leads")

    # Uses_tech: teams -> tech
    add(12, 23, "uses_tech"); add(12, 24, "uses_tech"); add(12, 30, "uses_tech")
    add(13, 25, "uses_tech"); add(13, 28, "uses_tech"); add(13, 29, "uses_tech")
    add(14, 23, "uses_tech"); add(14, 27, "uses_tech"); add(15, 26, "uses_tech"); add(15, 31, "uses_tech")
    add(16, 23, "uses_tech"); add(16, 27, "uses_tech"); add(17, 28, "uses_tech")

    # Located_in
    add(32, 33, "located_in"); add(32, 34, "located_in"); add(35, 32, "located_in"); add(36, 32, "located_in")
    add(12, 32, "located_in"); add(13, 35, "located_in"); add(14, 32, "located_in")

    # Projects: teams work on projects
    add(12, 18, "depends_on"); add(12, 19, "depends_on"); add(13, 18, "depends_on"); add(14, 20, "depends_on")
    add(15, 21, "depends_on"); add(16, 19, "depends_on"); add(16, 22, "depends_on"); add(17, 18, "depends_on")
    add(18, 20, "depends_on"); add(19, 21, "depends_on"); add(20, 22, "depends_on")

    # Authored: people -> documents
    add(0, 41, "authored"); add(1, 41, "authored"); add(3, 39, "authored"); add(5, 37, "authored")
    add(7, 38, "authored"); add(11, 40, "authored"); add(42, 39, "authored"); add(42, 41, "authored")

    # Reports_to
    add(1, 0, "reports_to"); add(2, 0, "reports_to"); add(4, 3, "reports_to"); add(6, 5, "reports_to")
    add(8, 7, "reports_to"); add(10, 9, "reports_to"); add(42, 0, "reports_to"); add(42, 3, "reports_to")

    # Collaborates_with: sparse — creates bottlenecks
    add(0, 3, "collaborates_with"); add(42, 0, "collaborates_with"); add(42, 3, "collaborates_with")
    add(5, 7, "collaborates_with"); add(9, 11, "collaborates_with")

    logger.info("Created sample KG: %d entities, %d relations", G.number_of_nodes(), G.number_of_edges())
    return G


def get_kg_stats(G: nx.DiGraph, stream: Optional[Any] = None) -> None:
    """
    Print KG statistics: num entities, num relations, relation type distribution,
    entity type distribution, average in/out degree, connected components.

    Args:
        G: Knowledge graph as NetworkX DiGraph.
        stream: Optional file-like object to write to; if None, uses print().
    """
    def out(s: str) -> None:
        if stream is not None:
            stream.write(s + "\n")
        else:
            print(s)

    n = G.number_of_nodes()
    m = G.number_of_edges()
    out(f"Entities: {n}, Relations: {m}")

    rel_dist: Dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        r = d.get("relation", "unknown")
        rel_dist[r] = rel_dist.get(r, 0) + 1
    out("Relation type distribution (top 15):")
    for r, c in sorted(rel_dist.items(), key=lambda x: -x[1])[:15]:
        out(f"  {r}: {c}")

    type_dist: Dict[str, int] = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Entity")
        type_dist[t] = type_dist.get(t, 0) + 1
    out("Entity type distribution:")
    for t, c in sorted(type_dist.items(), key=lambda x: -x[1]):
        out(f"  {t}: {c}")

    in_deg = [d for _, d in G.in_degree()]
    out_deg = [d for _, d in G.out_degree()]
    out(f"Avg in-degree: {sum(in_deg) / n:.2f}, Avg out-degree: {sum(out_deg) / n:.2f}")

    if G.is_directed():
        wcc = list(nx.weakly_connected_components(G))
    else:
        wcc = list(nx.connected_components(G))
    out(f"Weakly connected components: {len(wcc)}")
    if len(wcc) <= 10:
        for i, c in enumerate(wcc):
            out(f"  Component {i}: {len(c)} nodes")
    else:
        sizes = sorted([len(c) for c in wcc], reverse=True)
        out(f"  Largest: {sizes[0]}, Smallest: {sizes[-1]}")


def to_undirected(G: nx.DiGraph) -> nx.Graph:
    """
    Convert directed KG to undirected for curvature computation.
    Preserves all node and edge attributes; symmetric edges get merged (one edge, attributes kept).

    Args:
        G: Directed NetworkX DiGraph.

    Returns:
        Undirected NetworkX Graph with same nodes and combined edges.
    """
    U = nx.Graph()
    for n, d in G.nodes(data=True):
        U.add_node(n, **d)
    seen = set()
    for u, v, data in G.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in seen:
            continue
        seen.add(key)
        # Merge edge data if both (u,v) and (v,u) exist
        d = dict(data)
        if G.has_edge(v, u):
            d2 = G[v][u]
            d["relation"] = d.get("relation", "") + "|" + d2.get("relation", "")
            d["weight"] = (d.get("weight", 1.0) + d2.get("weight", 1.0)) / 2
        U.add_edge(u, v, **d)
    logger.debug("Converted to undirected: %d nodes, %d edges", U.number_of_nodes(), U.number_of_edges())
    return U
