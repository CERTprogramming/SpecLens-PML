"""
First DPG experiment for the SpecLens-PML Random Forest candidate.

This script builds Decision Predicate Graph (DPG) explanations for
``models/forest.pkl`` using the training context saved by ``pipeline/train.py``.
DPG is applied to the Random Forest candidate because DPG requires a tree-based
model; the operational champion selected by SpecLens-PML may still be logistic.

The experiment is intentionally separate from the core training, promotion, and
inference pipeline. It does not modify champion governance logic. Graph image
rendering is optional and disabled by default because large DPGs can be slow to
render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import argparse
import ast
import json
import multiprocessing as mp
import os
import signal
import shutil
import subprocess
import sys
import textwrap

import joblib
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DPG_ROOT = ROOT.parent / "DPG"

FOREST_MODEL_PATH = ROOT / "models" / "forest.pkl"
FOREST_CONTEXT_PATH = ROOT / "models" / "forest_training_context.pkl"

OUTPUT_DIR = ROOT / "experiments" / "dpg_outputs"

DEFAULT_RENDER_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RENDER_NODES = 200
DEFAULT_MAX_RENDER_EDGES = 1000
DEFAULT_SIMPLIFIED_TOP_K_NODES = 30
DEFAULT_SIMPLIFIED_NODE_METRIC = "betweenness"
DEFAULT_SIMPLIFIED_MAX_EDGES = 80
DEFAULT_COMMUNITY_MIXED_MARGIN = 0.10
DEFAULT_COMMUNITY_TOP_PREDICATES = 3

SIMPLIFIED_NODE_METRICS = {
    "betweenness": "Betweenness centrality",
    "local_reaching": "Local reaching centrality",
    "degree": "Degree",
}


# ---------------------------------------------------------------------------
# DPG Import
# ---------------------------------------------------------------------------

def load_dpg_explainer() -> Any:
    """
    Import ``DPGExplainer`` from the sibling DPG repository or environment.

    Returns
    -------
    Any
        The DPGExplainer class.

    Raises
    ------
    ImportError
        If DPG or one of its Python dependencies cannot be imported.
    """
    if DPG_ROOT.exists() and str(DPG_ROOT) not in sys.path:
        sys.path.insert(0, str(DPG_ROOT))

    try:
        from dpg import DPGExplainer
    except ImportError as exc:
        print("\nDPG could not be imported.")
        print(f"Expected sibling repository: {DPG_ROOT}")
        print("\nSetup options:")
        print("- Install DPG and its dependencies in the active environment:")
        print("  python3 -m pip install -e ../DPG")
        print("- Or run this script from an environment where the DPG package is available.")
        print(f"\nImport error: {exc}")
        raise

    return DPGExplainer


# ---------------------------------------------------------------------------
# Artifact Loading
# ---------------------------------------------------------------------------

def load_forest_model(path: Path) -> Any:
    """
    Load the trained Random Forest candidate model.

    Parameters
    ----------
    path : Path
        Path to ``models/forest.pkl``.

    Returns
    -------
    Any
        Trained tree-based model artifact.

    Raises
    ------
    FileNotFoundError
        If the model artifact is missing.
    ValueError
        If the loaded model is not tree-based enough for DPG.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    model = joblib.load(path)

    if not hasattr(model, "estimators_"):
        raise ValueError(
            "DPG requires a tree-based ensemble model, but forest.pkl does not "
            "expose estimators_."
        )

    return model


def load_training_context(path: Path) -> tuple[list[str], pd.DataFrame, pd.Series]:
    """
    Load and validate the DPG-ready training context sidecar.

    Parameters
    ----------
    path : Path
        Path to ``models/forest_training_context.pkl``.

    Returns
    -------
    tuple[list[str], pandas.DataFrame, pandas.Series]
        Feature names, training feature matrix, and training labels.

    Raises
    ------
    FileNotFoundError
        If the context artifact is missing.
    ValueError
        If required keys or dimensions are invalid.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    context = joblib.load(path)

    if not isinstance(context, dict):
        raise ValueError("forest_training_context.pkl must contain a dictionary.")

    required_keys = {"feature_names", "training_features", "training_labels"}
    missing_keys = required_keys - set(context)
    if missing_keys:
        raise ValueError(f"Training context is missing keys: {sorted(missing_keys)}")

    feature_names = list(context["feature_names"])
    training_features = context["training_features"]
    training_labels = context["training_labels"]

    X = pd.DataFrame(training_features).copy()
    y = pd.Series(training_labels).reset_index(drop=True)

    if X.empty:
        raise ValueError("Training feature matrix is empty.")

    if len(feature_names) != X.shape[1]:
        raise ValueError(
            "Feature name count does not match training feature matrix width."
        )

    if list(X.columns) != feature_names:
        if all(name in X.columns for name in feature_names):
            X = X[feature_names]
        else:
            X.columns = feature_names

    X = X.reset_index(drop=True)

    if len(X) != len(y):
        raise ValueError("Training feature and label row counts do not match.")

    return feature_names, X, y


# ---------------------------------------------------------------------------
# DPG Construction
# ---------------------------------------------------------------------------

def build_target_names(labels: pd.Series) -> list[str]:
    """
    Build readable class names for DPG class nodes.

    Parameters
    ----------
    labels : pandas.Series
        Training labels where 0 means SAFE and 1 means RISKY.

    Returns
    -------
    list[str]
        Target names in model class order.
    """
    unique_labels = sorted(labels.dropna().unique().tolist())

    if unique_labels == [0, 1]:
        return ["SAFE", "RISKY"]

    return [str(label) for label in unique_labels]


def build_dpg_explanation(
    DPGExplainer: Any,
    model: Any,
    feature_names: list[str],
    training_features: pd.DataFrame,
    training_labels: pd.Series,
) -> tuple[Any, Any]:
    """
    Build the global DPG explanation for the Random Forest candidate.

    Parameters
    ----------
    DPGExplainer : Any
        DPGExplainer class imported from DPG.
    model : Any
        Trained Random Forest candidate.
    feature_names : list[str]
        Feature names used during training.
    training_features : pandas.DataFrame
        Feature matrix used to train the saved candidate.
    training_labels : pandas.Series
        Training labels used to derive readable target names.

    Returns
    -------
    tuple[Any, Any]
        Fitted DPG explainer and global explanation object.
    """
    dpg_config = {
        "dpg": {
            "default": {
                "perc_var": 0.000000001,
                "decimal_threshold": 6,
                "n_jobs": 1,
            },
            "graph_construction": {
                "mode": "execution_trace",
            },
        }
    }

    explainer = DPGExplainer(
        model=model,
        feature_names=feature_names,
        target_names=build_target_names(training_labels),
        dpg_config=dpg_config,
    )

    explanation = explainer.explain_global(
        training_features.values,
        communities=True,
        community_threshold=0.2,
    )

    return explainer, explanation


# ---------------------------------------------------------------------------
# Output Helpers
# ---------------------------------------------------------------------------

def json_default(value: Any) -> Any:
    """
    Convert common scientific Python objects into JSON-friendly values.

    Parameters
    ----------
    value : Any
        Object passed by ``json.dumps`` when default serialization fails.

    Returns
    -------
    Any
        JSON-friendly fallback representation.
    """
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def write_json(path: Path, payload: Any) -> Path:
    """
    Write a JSON artifact with stable indentation.

    Parameters
    ----------
    path : Path
        Output path.
    payload : Any
        JSON-serializable payload, with fallback conversion.

    Returns
    -------
    Path
        The written output path.
    """
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n")
    return path


def save_global_summary(
    path: Path,
    model: Any,
    explanation: Any,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
) -> Path:
    """
    Save a compact text summary of the global DPG explanation.

    Parameters
    ----------
    path : Path
        Output text path.
    model : Any
        Trained Random Forest candidate.
    explanation : Any
        DPG global explanation object.
    training_features : pandas.DataFrame
        Training feature matrix.
    training_labels : pandas.Series
        Training labels.

    Returns
    -------
    Path
        The written output path.
    """
    label_counts = training_labels.value_counts().sort_index()
    class_counts = ", ".join(
        f"{label}: {count}" for label, count in label_counts.items()
    )

    lines = [
        "=== SpecLens-PML DPG Global Summary ===",
        "",
        f"Model artifact: {FOREST_MODEL_PATH.relative_to(ROOT)}",
        f"Training context: {FOREST_CONTEXT_PATH.relative_to(ROOT)}",
        "DPG target: Random Forest candidate model",
        "Governance note: champion promotion logic is unchanged.",
        "",
        f"Training samples: {len(training_features)}",
        f"Features: {training_features.shape[1]}",
        f"Training labels: {class_counts}",
        f"Forest estimators: {len(getattr(model, 'estimators_', []))}",
        "",
        f"DPG nodes: {explanation.graph.number_of_nodes()}",
        f"DPG edges: {explanation.graph.number_of_edges()}",
        f"Node metric rows: {len(explanation.node_metrics)}",
        f"Edge metric rows: {len(explanation.edge_metrics)}",
        f"Communities computed: {explanation.communities is not None}",
    ]

    path.write_text("\n".join(lines) + "\n")
    return path


def is_predicate_label(label: Any) -> bool:
    """
    Return True when a DPG node label looks like a decision predicate.

    Parameters
    ----------
    label : Any
        DPG node label.

    Returns
    -------
    bool
        Whether the label appears to describe a tree predicate.
    """
    text = str(label)
    return (
        ("<=" in text or ">" in text or "<" in text)
        and not text.startswith("Class ")
        and not text.startswith("Pred ")
    )


def append_top_metric_section(
    lines: list[str],
    title: str,
    predicates: pd.DataFrame,
    metric: str,
    top_n: int,
) -> None:
    """
    Append a top-predicate section sorted by a DPG node metric.

    Parameters
    ----------
    lines : list[str]
        Mutable output lines.
    title : str
        Section title.
    predicates : pandas.DataFrame
        Predicate node metrics.
    metric : str
        Metric column used for sorting.
    top_n : int
        Number of predicates to include.
    """
    lines.extend(["", title])

    if metric not in predicates.columns:
        lines.append(f"- Metric not available: {metric}")
        return

    top_rows = predicates.sort_values(metric, ascending=False).head(top_n)

    if top_rows.empty:
        lines.append("- No predicate nodes available.")
        return

    for _, row in top_rows.iterrows():
        lines.append(f"- {row['Label']} ({metric}: {float(row[metric]):.6f})")


def save_top_predicates(
    path: Path,
    node_metrics: pd.DataFrame,
    top_n: int = 10,
) -> Path:
    """
    Save a text summary of the most central DPG predicates.

    Parameters
    ----------
    path : Path
        Output text path.
    node_metrics : pandas.DataFrame
        DPG node metric table.
    top_n : int
        Number of predicates per section.

    Returns
    -------
    Path
        The written output path.
    """
    predicates = node_metrics[
        node_metrics["Label"].map(is_predicate_label)
    ].copy()

    lines = [
        "=== Top DPG Predicates ===",
        "",
        "These predicates come from the Random Forest candidate model.",
    ]

    append_top_metric_section(
        lines,
        "Top predicates by local reaching centrality:",
        predicates,
        "Local reaching centrality",
        top_n,
    )
    append_top_metric_section(
        lines,
        "Top predicates by betweenness centrality:",
        predicates,
        "Betweenness centrality",
        top_n,
    )
    append_top_metric_section(
        lines,
        "Most connected predicates by degree:",
        predicates,
        "Degree",
        top_n,
    )

    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Local Explanations
# ---------------------------------------------------------------------------

def sample_index_for_label(labels: pd.Series, label: int) -> Optional[int]:
    """
    Return the first training sample index for a label.

    Parameters
    ----------
    labels : pandas.Series
        Training labels.
    label : int
        Desired label value.

    Returns
    -------
    Optional[int]
        First matching index, or None if no sample exists.
    """
    matches = labels[labels == label]
    if matches.empty:
        return None
    return int(matches.index[0])


def save_local_explanation(
    explainer: Any,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    sample_name: str,
    sample_index: int,
    output_dir: Path,
) -> list[Path]:
    """
    Save local DPG explanation artifacts for one training sample.

    Parameters
    ----------
    explainer : Any
        Fitted DPG explainer.
    training_features : pandas.DataFrame
        Training feature matrix.
    training_labels : pandas.Series
        Training labels.
    sample_name : str
        Human-readable sample name used in output filenames.
    sample_index : int
        Row index to explain.
    output_dir : Path
        Output directory.

    Returns
    -------
    list[Path]
        Written output paths.
    """
    sample = training_features.iloc[sample_index]
    local = explainer.explain_local(
        sample=sample.values,
        sample_id=sample_index,
    )

    local_table = explainer.local_path_dataframe(local)
    local_table_path = output_dir / f"local_{sample_name}_paths.csv"
    local_table.to_csv(local_table_path, index=False)

    summary_path = output_dir / f"local_{sample_name}_summary.json"
    write_json(
        summary_path,
        {
            "sample_id": sample_index,
            "true_label": int(training_labels.iloc[sample_index]),
            "majority_vote": local.majority_vote,
            "class_votes": local.class_votes,
            "graph_validated": local.graph_validated,
            "all_trees_valid": local.all_trees_valid,
            "path_mode": local.path_mode,
            "sample_confidence": local.sample_confidence,
        },
    )

    return [local_table_path, summary_path]


def save_representative_local_explanations(
    explainer: Any,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    output_dir: Path,
) -> list[Path]:
    """
    Save local explanations for one SAFE and one RISKY training sample.

    Parameters
    ----------
    explainer : Any
        Fitted DPG explainer.
    training_features : pandas.DataFrame
        Training feature matrix.
    training_labels : pandas.Series
        Training labels.
    output_dir : Path
        Output directory.

    Returns
    -------
    list[Path]
        Written output paths.
    """
    outputs: list[Path] = []
    samples = [
        ("safe", 0),
        ("risky", 1),
    ]

    for sample_name, label in samples:
        sample_index = sample_index_for_label(training_labels, label)
        if sample_index is None:
            print(f"No {sample_name.upper()} training sample found. Skipping local explanation.")
            continue

        outputs.extend(
            save_local_explanation(
                explainer,
                training_features,
                training_labels,
                sample_name,
                sample_index,
                output_dir,
            )
        )

    return outputs


# ---------------------------------------------------------------------------
# Artifact Saving
# ---------------------------------------------------------------------------

def save_structured_artifacts(
    model: Any,
    explanation: Any,
    explainer: Any,
    training_features: pd.DataFrame,
    training_labels: pd.Series,
    output_dir: Path,
) -> list[Path]:
    """
    Save DPG explanation outputs in text, JSON, and CSV form.

    Parameters
    ----------
    model : Any
        Trained Random Forest candidate.
    explanation : Any
        DPG global explanation object.
    explainer : Any
        Fitted DPG explainer.
    training_features : pandas.DataFrame
        Training feature matrix.
    training_labels : pandas.Series
        Training labels.
    output_dir : Path
        Output directory.

    Returns
    -------
    list[Path]
        Written output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    outputs.append(
        save_global_summary(
            output_dir / "global_summary.txt",
            model,
            explanation,
            training_features,
            training_labels,
        )
    )

    nodes_path = output_dir / "dpg_nodes.csv"
    pd.DataFrame(explanation.nodes, columns=["node_id", "label"]).to_csv(
        nodes_path,
        index=False,
    )
    outputs.append(nodes_path)

    node_metrics_path = output_dir / "dpg_node_metrics.csv"
    explanation.node_metrics.to_csv(node_metrics_path, index=False)
    outputs.append(node_metrics_path)

    edge_metrics_path = output_dir / "dpg_edge_metrics.csv"
    explanation.edge_metrics.to_csv(edge_metrics_path, index=False)
    outputs.append(edge_metrics_path)

    outputs.append(
        write_json(
            output_dir / "dpg_class_boundaries.json",
            explanation.class_boundaries,
        )
    )

    if explanation.communities is not None:
        communities_path = output_dir / "dpg_communities.csv"
        try:
            from metrics.graph import GraphMetrics

            GraphMetrics.communities_to_csv(
                explanation.communities,
                str(communities_path),
            )
            outputs.append(communities_path)
        except Exception as exc:
            print("\nCommunity CSV export failed.")
            print(f"Reason: {exc}")
            outputs.append(
                write_json(
                    output_dir / "dpg_communities.json",
                    explanation.communities,
                )
            )

    outputs.append(
        save_top_predicates(
            output_dir / "top_predicates.txt",
            explanation.node_metrics,
        )
    )

    outputs.extend(
        save_representative_local_explanations(
            explainer,
            training_features,
            training_labels,
            output_dir,
        )
    )

    return outputs


# ---------------------------------------------------------------------------
# Simplified Global Rendering
# ---------------------------------------------------------------------------

def load_metric_tables(
    output_dir: Path,
    explanation: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load DPG metric tables from disk when available.

    Parameters
    ----------
    output_dir : Path
        Directory containing generated DPG artifacts.
    explanation : Any
        In-memory DPG explanation used as a fallback.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Node metrics and edge metrics.
    """
    node_metrics_path = output_dir / "dpg_node_metrics.csv"
    edge_metrics_path = output_dir / "dpg_edge_metrics.csv"

    if node_metrics_path.exists():
        node_metrics = pd.read_csv(node_metrics_path)
    else:
        node_metrics = explanation.node_metrics.copy()

    if edge_metrics_path.exists():
        edge_metrics = pd.read_csv(edge_metrics_path)
    else:
        edge_metrics = explanation.edge_metrics.copy()

    return node_metrics, edge_metrics


def is_class_node_label(label: Any) -> bool:
    """
    Return whether a DPG node label represents a problem class leaf.

    Parameters
    ----------
    label : Any
        Node label to inspect.

    Returns
    -------
    bool
        ``True`` when the normalized label starts with ``"Class "``.
    """
    return str(label).strip().lower().startswith("class ")


def _sort_edges_by_weight(edge_table: pd.DataFrame) -> pd.DataFrame:
    """
    Return edges ordered by descending weight.

    Rows with the same weight are sorted deterministically by source and target
    identifiers.

    Parameters
    ----------
    edge_table : pandas.DataFrame
        DPG edge table to sort.

    Returns
    -------
    pandas.DataFrame
        Sorted copy of the edge table.
    """
    ordered = edge_table.copy()
    if "Weight" in ordered.columns:
        ordered["Weight"] = pd.to_numeric(
            ordered["Weight"],
            errors="coerce",
        ).fillna(0.0)
        ordered = ordered.sort_values(
            ["Weight", "Source_id", "Target_id"],
            ascending=[False, True, True],
        )
    else:
        ordered = ordered.sort_values(
            ["Source_id", "Target_id"],
            ascending=[True, True],
        )
    return ordered


def select_simplified_global_tables(
    node_metrics: pd.DataFrame,
    edge_metrics: pd.DataFrame,
    metric_name: str,
    top_k_nodes: int,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select a readable class-aware subgraph of the global DPG.

    The ``top_k_nodes`` limit applies to predicate nodes. All class leaves are
    retained independently of their centrality score. For every class leaf,
    the strongest incoming edge is preserved; when its source predicate is not
    already among the top-k predicates, that predicate is added as a bridge.
    Remaining edges are selected by descending weight up to ``max_edges``.

    Parameters
    ----------
    node_metrics : pandas.DataFrame
        Full DPG node metric table.
    edge_metrics : pandas.DataFrame
        Full DPG edge metric table.
    metric_name : str
        Friendly metric key used for predicate-node ranking.
    top_k_nodes : int
        Number of high-importance predicate nodes to retain. Class leaves and
        any bridge predicates required to connect them are additional.
    max_edges : int
        Preferred maximum number of retained edges. One incoming edge per
        class is always retained, even when this requires exceeding the limit.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Simplified node table and edge table.

    Raises
    ------
    ValueError
        If required columns or requested metric are unavailable.
    """
    metric_column = SIMPLIFIED_NODE_METRICS[metric_name]
    required_node_columns = {"Node", "Label", metric_column}
    missing_node_columns = required_node_columns - set(node_metrics.columns)
    if missing_node_columns:
        raise ValueError(
            f"Node metrics are missing columns: {sorted(missing_node_columns)}"
        )

    required_edge_columns = {"Source_id", "Target_id"}
    missing_edge_columns = required_edge_columns - set(edge_metrics.columns)
    if missing_edge_columns:
        raise ValueError(
            f"Edge metrics are missing columns: {sorted(missing_edge_columns)}"
        )

    if top_k_nodes <= 0:
        raise ValueError("--top-k-nodes must be positive.")

    if max_edges < 0:
        raise ValueError("--max-edges must be zero or positive.")

    ranked_nodes = node_metrics.copy()
    ranked_nodes["Node"] = ranked_nodes["Node"].astype(str)
    ranked_nodes[metric_column] = pd.to_numeric(
        ranked_nodes[metric_column],
        errors="coerce",
    ).fillna(0.0)
    ranked_nodes["_is_class"] = ranked_nodes["Label"].map(is_class_node_label)

    class_nodes = ranked_nodes[ranked_nodes["_is_class"]].copy()
    predicate_nodes = ranked_nodes[~ranked_nodes["_is_class"]].copy()
    predicate_nodes = predicate_nodes.sort_values(
        [metric_column, "Label"],
        ascending=[False, True],
    )

    simplified_nodes = predicate_nodes.head(top_k_nodes).copy()
    selected_ids = set(simplified_nodes["Node"])
    class_ids = set(class_nodes["Node"])

    normalized_edges = edge_metrics.copy()
    normalized_edges["Source_id"] = normalized_edges["Source_id"].astype(str)
    normalized_edges["Target_id"] = normalized_edges["Target_id"].astype(str)
    normalized_edges = _sort_edges_by_weight(normalized_edges)

    # Keep every class visible and connected to its strongest predecessor.
    mandatory_class_edges: list[pd.DataFrame] = []
    bridge_ids: set[str] = set()
    for class_id in sorted(class_ids):
        incoming = normalized_edges[normalized_edges["Target_id"] == class_id]
        if incoming.empty:
            continue

        incoming_from_selected = incoming[incoming["Source_id"].isin(selected_ids)]
        chosen = (
            incoming_from_selected.head(1)
            if not incoming_from_selected.empty
            else incoming.head(1)
        )
        mandatory_class_edges.append(chosen)
        bridge_ids.update(chosen["Source_id"].astype(str))

    if bridge_ids:
        bridge_nodes = predicate_nodes[predicate_nodes["Node"].isin(bridge_ids)]
        simplified_nodes = pd.concat(
            [simplified_nodes, bridge_nodes],
            ignore_index=True,
        ).drop_duplicates(subset=["Node"], keep="first")
        selected_ids.update(bridge_ids)

    simplified_nodes = pd.concat(
        [simplified_nodes, class_nodes],
        ignore_index=True,
    ).drop_duplicates(subset=["Node"], keep="first")
    selected_ids.update(class_ids)

    candidate_edges = normalized_edges[
        normalized_edges["Source_id"].isin(selected_ids)
        & normalized_edges["Target_id"].isin(selected_ids)
    ].copy()

    if mandatory_class_edges:
        mandatory_edges = pd.concat(mandatory_class_edges, ignore_index=True)
        mandatory_edges = mandatory_edges.drop_duplicates(
            subset=["Source_id", "Target_id"],
            keep="first",
        )
    else:
        mandatory_edges = candidate_edges.head(0).copy()

    mandatory_pairs = set(
        zip(mandatory_edges["Source_id"], mandatory_edges["Target_id"])
    )
    optional_mask = [
        (source, target) not in mandatory_pairs
        for source, target in zip(
            candidate_edges["Source_id"],
            candidate_edges["Target_id"],
        )
    ]
    optional_edges = candidate_edges.loc[optional_mask]

    optional_limit = max(0, max_edges - len(mandatory_edges))
    simplified_edges = pd.concat(
        [mandatory_edges, optional_edges.head(optional_limit)],
        ignore_index=True,
    ).drop_duplicates(subset=["Source_id", "Target_id"], keep="first")

    # Class connectivity is more important than a strict edge cap.
    if len(mandatory_edges) > max_edges:
        simplified_edges = mandatory_edges.copy()

    simplified_nodes = simplified_nodes.drop(columns=["_is_class"], errors="ignore")
    return simplified_nodes, simplified_edges


def dot_quote(value: Any) -> str:
    """
    Quote a value for DOT output.

    Parameters
    ----------
    value : Any
        Value to render as a DOT string.

    Returns
    -------
    str
        Escaped DOT string literal.
    """
    text = str(value)
    text = text.replace("\\", "\\\\").replace('"', '\\"')
    text = text.replace("\n", "\\n")
    return f'"{text}"'


def wrap_dot_label(label: Any, width: int = 32) -> str:
    """
    Wrap long node labels for a compact DOT graph.

    Parameters
    ----------
    label : Any
        Original DPG node label.
    width : int
        Preferred line width.

    Returns
    -------
    str
        Wrapped label.
    """
    return "\n".join(textwrap.wrap(str(label), width=width)) or str(label)


def write_simplified_dot(
    path: Path,
    simplified_nodes: pd.DataFrame,
    simplified_edges: pd.DataFrame,
    metric_name: str,
) -> Path:
    """
    Write a simplified global DPG as a DOT file.

    Parameters
    ----------
    path : Path
        DOT output path.
    simplified_nodes : pandas.DataFrame
        Retained node table.
    simplified_edges : pandas.DataFrame
        Retained edge table.
    metric_name : str
        Friendly metric key used for node ranking.

    Returns
    -------
    Path
        Written DOT path.
    """
    metric_column = SIMPLIFIED_NODE_METRICS[metric_name]
    lines = [
        "digraph SimplifiedDPG {",
        "  graph [rankdir=LR, bgcolor=\"white\", overlap=false, splines=true];",
        (
            '  node [shape=box, style="rounded,filled", '
            'fontname="Helvetica", fontsize=10, margin="0.08,0.05"];'
        ),
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#6b7280\", arrowsize=0.7];",
        f"  labelloc=\"t\";",
        (
            "  label="
            f"{dot_quote('SpecLens-PML simplified DPG: central predicates and '
                         f'class leaves ({metric_name})')};"
        ),
        "",
    ]

    for _, row in simplified_nodes.iterrows():
        node_id = row["Node"]
        label = wrap_dot_label(row["Label"])
        metric_value = float(row.get(metric_column, 0.0))
        raw_label = str(row["Label"])
        is_class = is_class_node_label(raw_label)

        if is_class:
            normalized_label = raw_label.upper()
            if "SAFE" in normalized_label:
                fillcolor = "#bbf7d0"
                border_color = "#166534"
            elif "RISKY" in normalized_label:
                fillcolor = "#fecaca"
                border_color = "#991b1b"
            else:
                fillcolor = "#bfdbfe"
                border_color = "#1d4ed8"
            display_label = label
            shape = "doubleoctagon"
            penwidth = 2.4
        else:
            fillcolor = "#fef3c7"
            border_color = "#374151"
            display_label = f"{label}\n{metric_name}: {metric_value:.4f}"
            shape = "box"
            penwidth = 1.0

        lines.append(
            "  "
            f"{dot_quote(node_id)} "
            "["
            f"label={dot_quote(display_label)}, "
            f"fillcolor={dot_quote(fillcolor)}, "
            f"color={dot_quote(border_color)}, "
            f"shape={dot_quote(shape)}, "
            f"penwidth={penwidth:.1f}"
            "];"
        )

    lines.append("")

    for _, row in simplified_edges.iterrows():
        edge_attrs = []
        if "Weight" in simplified_edges.columns:
            edge_label = f"w={float(row['Weight']):.0f}"
            edge_attrs.append(f"label={dot_quote(edge_label)}")
            edge_attrs.append(
                f"penwidth={max(1.0, min(5.0, 1.0 + float(row['Weight']) / 20.0)):.2f}"
            )

        attr_text = f" [{', '.join(edge_attrs)}]" if edge_attrs else ""
        lines.append(
            "  "
            f"{dot_quote(row['Source_id'])} -> {dot_quote(row['Target_id'])}"
            f"{attr_text};"
        )

    lines.append("}")
    path.write_text("\n".join(lines) + "\n")
    return path


def render_dot_file(
    dot_path: Path,
    output_dir: Path,
    timeout_seconds: int,
) -> list[Path]:
    """
    Render a DOT file to PNG and SVG with Graphviz if available.

    Parameters
    ----------
    dot_path : Path
        DOT source path.
    output_dir : Path
        Output directory.
    timeout_seconds : int
        Maximum seconds allowed per Graphviz render.

    Returns
    -------
    list[Path]
        Rendered image paths, if Graphviz succeeds.
    """
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        print(
            "Graphviz 'dot' command not found. "
            "Simplified DOT was saved without image rendering."
        )
        return []

    rendered_paths: list[Path] = []

    for fmt in ["png", "svg"]:
        out_path = output_dir / f"simplified_global_dpg.{fmt}"
        try:
            subprocess.run(
                [
                    dot_binary,
                    f"-T{fmt}",
                    str(dot_path),
                    "-o",
                    str(out_path),
                ],
                check=True,
                timeout=timeout_seconds,
            )
            rendered_paths.append(out_path)
        except subprocess.TimeoutExpired:
            print(f"Simplified {fmt.upper()} rendering timed out after {timeout_seconds} seconds.")
        except subprocess.CalledProcessError as exc:
            print(f"Simplified {fmt.upper()} rendering failed: {exc}")

    return rendered_paths


def save_simplified_global_rendering(
    output_dir: Path,
    explanation: Any,
    top_k_nodes: int,
    metric_name: str,
    max_edges: int,
    timeout_seconds: int,
) -> list[Path]:
    """
    Save and render a simplified global DPG view.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    explanation : Any
        In-memory DPG explanation used as a fallback source.
    top_k_nodes : int
        Number of high-importance nodes to retain.
    metric_name : str
        Friendly metric key used for node ranking.
    max_edges : int
        Maximum number of retained edges.
    timeout_seconds : int
        Maximum seconds per Graphviz render.

    Returns
    -------
    list[Path]
        Written DOT, CSV, and rendered image paths.
    """
    print("\n=== Simplified global rendering ===")

    node_metrics, edge_metrics = load_metric_tables(output_dir, explanation)
    simplified_nodes, simplified_edges = select_simplified_global_tables(
        node_metrics,
        edge_metrics,
        metric_name,
        top_k_nodes,
        max_edges,
    )

    outputs: list[Path] = []

    nodes_path = output_dir / "simplified_global_nodes.csv"
    simplified_nodes.to_csv(nodes_path, index=False)
    outputs.append(nodes_path)

    edges_path = output_dir / "simplified_global_edges.csv"
    simplified_edges.to_csv(edges_path, index=False)
    outputs.append(edges_path)

    dot_path = write_simplified_dot(
        output_dir / "simplified_global_dpg.dot",
        simplified_nodes,
        simplified_edges,
        metric_name,
    )
    outputs.append(dot_path)

    print(
        "Simplified graph selected "
        f"{len(simplified_nodes)} nodes and {len(simplified_edges)} edges."
    )

    outputs.extend(render_dot_file(dot_path, output_dir, timeout_seconds))

    return outputs


# ---------------------------------------------------------------------------
# Class-aware Community Analysis
# ---------------------------------------------------------------------------

def _find_column(columns: list[str], candidates: tuple[str, ...]) -> Optional[str]:
    """
    Find a column whose normalized name matches one of the candidates.

    Parameters
    ----------
    columns : list[str]
        Available column names.
    candidates : tuple[str, ...]
        Accepted normalized aliases, checked in order.

    Returns
    -------
    Optional[str]
        Original column name when a match is found, otherwise ``None``.
    """
    normalized = {str(column).strip().lower().replace(" ", "_"): column for column in columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in normalized:
            return normalized[key]
    return None


def load_community_membership(output_dir: Path, explanation: Any) -> pd.DataFrame:
    """
    Load and normalize DPG community membership.

    DPG's ``GraphMetrics.communities_to_csv`` writes a long-form CSV with the
    columns ``Section``, ``Key``, and ``Value``. Community membership is stored
    in rows whose section is ``Clusters``; ``Key`` is the class-oriented cluster
    name and ``Value`` is a serialized list of predicate labels.

    Older or alternative two-column exports are also accepted. Predicate
    labels are mapped back to the node identifiers used by ``node_metrics`` so
    the rest of the analysis can join membership, node, and edge tables safely.

    Parameters
    ----------
    output_dir : Path
        Directory containing ``dpg_communities.csv`` and related artifacts.
    explanation : Any
        In-memory DPG explanation used as a fallback source.

    Returns
    -------
    pandas.DataFrame
        Normalized table with ``Node`` and ``Community`` columns.

    Raises
    ------
    ValueError
        If node metrics are invalid or community membership cannot be
        normalized from either the CSV artifact or the in-memory explanation.
    """

    node_metrics = getattr(explanation, "node_metrics", None)
    if not isinstance(node_metrics, pd.DataFrame):
        node_metrics = pd.DataFrame(node_metrics)
    if (
        node_metrics.empty
        or "Node" not in node_metrics.columns
        or "Label" not in node_metrics.columns
    ):
        raise ValueError("DPG node metrics must contain Node and Label columns.")

    node_table = node_metrics[["Node", "Label"]].copy()
    node_table["Node"] = node_table["Node"].astype(str)
    node_table["Label"] = node_table["Label"].astype(str)

    label_to_nodes: dict[str, list[str]] = {}
    for _, row in node_table.iterrows():
        label_to_nodes.setdefault(row["Label"], []).append(row["Node"])
    known_node_ids = set(node_table["Node"])

    def parse_members(value: Any) -> list[str]:
        """
        Normalize a serialized DPG cluster value.

        Parameters
        ----------
        value : Any
            Serialized member collection, mapping, scalar, or iterable.

        Returns
        -------
        list[str]
            Community member labels or node identifiers.
        """
        if isinstance(value, dict):
            return [str(item) for item in value.keys()]
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]
        if pd.isna(value):
            return []

        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            # Graceful fallback for a single, non-serialized value.
            return [text]
        if isinstance(parsed, dict):
            return [str(item) for item in parsed.keys()]
        if isinstance(parsed, (list, tuple, set)):
            return [str(item) for item in parsed]
        return [str(parsed)]

    def rows_from_mapping(mapping: Any) -> list[dict[str, Any]]:
        """
        Convert a community mapping to normalized membership rows.

        Parameters
        ----------
        mapping : Any
            Expected ``community -> members`` mapping.

        Returns
        -------
        list[dict[str, Any]]
            Rows containing normalized ``Node`` and ``Community`` values.
        """
        rows: list[dict[str, Any]] = []
        if not isinstance(mapping, dict):
            return rows
        for community_id, members_value in mapping.items():
            for member in parse_members(members_value):
                if member in known_node_ids:
                    rows.append({"Node": member, "Community": community_id})
                    continue
                for node_id in label_to_nodes.get(member, []):
                    rows.append({"Node": node_id, "Community": community_id})
        return rows

    path = output_dir / "dpg_communities.csv"
    csv_columns: list[str] = []
    csv_sections: list[str] = []
    if path.exists():
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        csv_columns = [str(column) for column in raw.columns]

        # Current DPG export: Section, Key, Value.
        section_col = _find_column(csv_columns, ("section",))
        key_col = _find_column(csv_columns, ("key",))
        value_col = _find_column(csv_columns, ("value",))
        if section_col and key_col and value_col:
            section_values = raw[section_col].astype(str).str.strip()
            csv_sections = sorted(section_values.unique().tolist())
            cluster_rows = raw[section_values.str.casefold() == "clusters".casefold()]
            rows: list[dict[str, Any]] = []
            for _, row in cluster_rows.iterrows():
                community_id = str(row[key_col]).strip()
                for member in parse_members(row[value_col]):
                    if member in known_node_ids:
                        rows.append({"Node": member, "Community": community_id})
                        continue
                    for node_id in label_to_nodes.get(member, []):
                        rows.append({"Node": node_id, "Community": community_id})
            if rows:
                return pd.DataFrame(rows).drop_duplicates()

        # Backwards-compatible two-column membership export.
        node_col = _find_column(
            csv_columns,
            ("node", "node_id", "id", "vertex", "predicate_id"),
        )
        community_col = _find_column(
            csv_columns,
            ("community", "community_id", "cluster", "cluster_id", "group"),
        )
        if node_col and community_col:
            result = raw[[node_col, community_col]].copy()
            result.columns = ["Node", "Community"]
            result["Node"] = result["Node"].astype(str)
            return result.drop_duplicates()

    communities = getattr(explanation, "communities", None)
    rows: list[dict[str, Any]] = []
    if isinstance(communities, dict):
        # Current DPG in-memory format.
        clusters = communities.get("Clusters")
        if clusters is not None:
            rows = rows_from_mapping(clusters)
        else:
            # Older direct ``community -> members`` format.
            rows = rows_from_mapping(communities)
    elif isinstance(communities, (list, tuple)):
        for community_id, members in enumerate(communities):
            for member in parse_members(members):
                if member in known_node_ids:
                    rows.append({"Node": member, "Community": community_id})
                    continue
                for node_id in label_to_nodes.get(member, []):
                    rows.append({"Node": node_id, "Community": community_id})

    if not rows:
        details = f"CSV columns={csv_columns}"
        if csv_sections:
            details += f", sections={csv_sections}"
        raise ValueError(
            "Could not normalize DPG communities. "
            f"{details}. Expected current DPG long format "
            "(Section, Key, Value) with a Clusters section, or a two-column "
            "node/community membership table."
        )
    return pd.DataFrame(rows).drop_duplicates()


def _community_display_label(position: int, dominant: str) -> str:
    """
    Build a reader-friendly community label.

    Parameters
    ----------
    position : int
        One-based display position of the community.
    dominant : str
        Dominant association label such as ``SAFE``, ``RISKY``, ``MIXED``, or
        ``UNCONNECTED``.

    Returns
    -------
    str
        Human-readable label used in reports and plots.
    """
    status = {
        "SAFE": "SAFE-dominant",
        "RISKY": "RISKY-dominant",
        "MIXED": "mixed",
        "UNCONNECTED": "unconnected",
    }.get(str(dominant).upper(), str(dominant).lower())
    return f"Community {position} — {status}"


def _top_predicate_text(
    community_nodes: pd.DataFrame,
    metric_column: str,
    limit: int,
) -> str:
    """
    Build a compact list of representative predicates.

    Parameters
    ----------
    community_nodes : pandas.DataFrame
        Predicate nodes belonging to one community.
    metric_column : str
        Node metric used for ranking.
    limit : int
        Maximum number of predicates to include.

    Returns
    -------
    str
        Semicolon-separated predicate labels, or an empty string when the
        requested metric is unavailable.
    """
    if metric_column not in community_nodes.columns or community_nodes.empty:
        return ""
    ranked = community_nodes.copy()
    ranked[metric_column] = pd.to_numeric(
        ranked[metric_column], errors="coerce"
    ).fillna(0.0)
    ranked = ranked.sort_values(
        [metric_column, "Label"], ascending=[False, True]
    )
    return "; ".join(ranked.head(limit)["Label"].astype(str).tolist())


def build_community_class_tables(
    node_metrics: pd.DataFrame,
    edge_metrics: pd.DataFrame,
    membership: pd.DataFrame,
    mixed_margin: float,
    top_predicates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate DPG communities and estimate their class association.

    Association is descriptive, not causal. It is computed from the total
    weight of direct outgoing edges from predicates in each community to the
    SAFE and RISKY class leaves.

    Parameters
    ----------
    node_metrics : pandas.DataFrame
        DPG node metrics, including node identifiers and labels.
    edge_metrics : pandas.DataFrame
        DPG edge metrics, including source, target, and optional weights.
    membership : pandas.DataFrame
        Normalized ``Node, Community`` membership table.
    mixed_margin : float
        Maximum absolute SAFE/RISKY score difference classified as ``MIXED``.
    top_predicates : int
        Number of representative predicates stored for each metric.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Community summary table and per-predicate community table.

    Raises
    ------
    ValueError
        If the configuration is invalid or no predicate nodes can be matched
        to communities.
    """
    if not 0.0 <= mixed_margin <= 1.0:
        raise ValueError("--community-mixed-margin must be between 0 and 1.")
    if top_predicates <= 0:
        raise ValueError("--community-top-predicates must be positive.")

    nodes = node_metrics.copy()
    nodes["Node"] = nodes["Node"].astype(str)
    nodes["Label"] = nodes["Label"].astype(str)
    membership = membership.copy()
    membership["Node"] = membership["Node"].astype(str)

    predicate_nodes = nodes[~nodes["Label"].map(is_class_node_label)].copy()
    class_nodes = nodes[nodes["Label"].map(is_class_node_label)].copy()
    joined = membership.merge(predicate_nodes, on="Node", how="inner")
    if joined.empty:
        raise ValueError("No predicate nodes could be matched to DPG communities.")

    edges = edge_metrics.copy()
    edges["Source_id"] = edges["Source_id"].astype(str)
    edges["Target_id"] = edges["Target_id"].astype(str)
    if "Weight" not in edges.columns:
        edges["Weight"] = 1.0
    edges["Weight"] = pd.to_numeric(edges["Weight"], errors="coerce").fillna(0.0)

    class_map = dict(zip(class_nodes["Node"], class_nodes["Label"]))
    class_edges = edges[edges["Target_id"].isin(class_map)].copy()
    class_edges["ClassLabel"] = class_edges["Target_id"].map(class_map)
    class_edges = class_edges.merge(
        membership.rename(columns={"Node": "Source_id"}),
        on="Source_id",
        how="inner",
    )

    predicate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for community_id, group in joined.groupby("Community", sort=True):
        group = group.drop_duplicates(subset=["Node"])
        community_edges = class_edges[class_edges["Community"] == community_id]
        safe_weight = float(
            community_edges.loc[
                community_edges["ClassLabel"].str.upper().str.contains("SAFE"),
                "Weight",
            ].sum()
        )
        risky_weight = float(
            community_edges.loc[
                community_edges["ClassLabel"].str.upper().str.contains("RISKY"),
                "Weight",
            ].sum()
        )
        total = safe_weight + risky_weight
        safe_score = safe_weight / total if total else 0.0
        risky_score = risky_weight / total if total else 0.0
        if total == 0:
            dominant = "UNCONNECTED"
        elif abs(safe_score - risky_score) <= mixed_margin:
            dominant = "MIXED"
        elif safe_score > risky_score:
            dominant = "SAFE"
        else:
            dominant = "RISKY"

        top_betweenness = _top_predicate_text(
            group, "Betweenness centrality", top_predicates
        )
        top_local = _top_predicate_text(
            group, "Local reaching centrality", top_predicates
        )
        top_degree = _top_predicate_text(group, "Degree", top_predicates)

        summary_rows.append(
            {
                "Community": community_id,
                "Predicate count": len(group),
                "SAFE edge weight": safe_weight,
                "RISKY edge weight": risky_weight,
                "SAFE association score": safe_score,
                "RISKY association score": risky_score,
                "Dominant class": dominant,
                "Top by betweenness": top_betweenness,
                "Top by local reaching": top_local,
                "Top by degree": top_degree,
            }
        )

        for _, row in group.iterrows():
            predicate_rows.append(
                {
                    "Community": community_id,
                    "Node": row["Node"],
                    "Label": row["Label"],
                    "Degree": row.get("Degree", 0.0),
                    "Betweenness centrality": row.get("Betweenness centrality", 0.0),
                    "Local reaching centrality": row.get("Local reaching centrality", 0.0),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["Dominant class", "Predicate count", "Community"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    summary.insert(
        1,
        "Display label",
        [
            _community_display_label(position, dominant)
            for position, dominant in enumerate(summary["Dominant class"], start=1)
        ],
    )
    predicates = pd.DataFrame(predicate_rows).sort_values(
        ["Community", "Betweenness centrality"], ascending=[True, False]
    )
    return summary, predicates


def write_community_summary_text(path: Path, summary: pd.DataFrame) -> Path:
    """
    Write a readable class-aware community report.

    Parameters
    ----------
    path : Path
        Output text path.
    summary : pandas.DataFrame
        Community summary generated by ``build_community_class_tables``.

    Returns
    -------
    Path
        Written output path.
    """
    lines = [
        "SpecLens-PML DPG community analysis",
        "====================================",
        "",
        "Class association is based on weighted direct edges from community",
        "predicates to class leaves. It describes structural association and",
        "must not be interpreted as causality.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.extend(
            [
                str(row["Display label"]),
                f"  DPG cluster: {row['Community']}",
                f"  Predicates: {int(row['Predicate count'])}",
                f"  Class association: SAFE {row['SAFE association score'] * 100:.1f}% | "
                f"RISKY {row['RISKY association score'] * 100:.1f}%",
                f"  SAFE edge weight: {row['SAFE edge weight']:.1f}",
                f"  RISKY edge weight: {row['RISKY edge weight']:.1f}",
                f"  Top betweenness: {row['Top by betweenness'] or 'n/a'}",
                f"  Top local reaching: {row['Top by local reaching'] or 'n/a'}",
                f"  Top degree: {row['Top by degree'] or 'n/a'}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_community_dot(path: Path, summary: pd.DataFrame) -> Path:
    """
    Write a compact class-aware community graph in DOT format.

    The graph contains one node per community and explicit ``Class SAFE`` and
    ``Class RISKY`` leaves. Community-to-class edges show normalized
    association percentages and aggregate weights.

    Parameters
    ----------
    path : Path
        DOT output path.
    summary : pandas.DataFrame
        Community summary generated by ``build_community_class_tables``.

    Returns
    -------
    Path
        Written DOT path.
    """
    styles = {
        "SAFE": ("#dcfce7", "#166534"),
        "RISKY": ("#fee2e2", "#991b1b"),
        "MIXED": ("#fef3c7", "#92400e"),
        "UNCONNECTED": ("#e5e7eb", "#4b5563"),
    }
    max_count = max(int(summary["Predicate count"].max()), 1)
    lines = [
        "digraph CommunityDPG {",
        '  graph [rankdir=LR, bgcolor="white", overlap=false, splines=true];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9, color="#6b7280", arrowsize=0.7];',
        '  labelloc="t";',
        f"  label={dot_quote('SpecLens-PML DPG: class-aware community summary')};",
        (
            '  "Class SAFE" [shape="doubleoctagon", fillcolor="#bbf7d0", '
            'color="#166534", penwidth=2.4];'
        ),
        (
            '  "Class RISKY" [shape="doubleoctagon", fillcolor="#fecaca", '
            'color="#991b1b", penwidth=2.4];'
        ),
        "",
    ]
    for _, row in summary.iterrows():
        dominant = str(row["Dominant class"])
        fill, border = styles.get(dominant, styles["UNCONNECTED"])
        top = str(row["Top by betweenness"] or row["Top by local reaching"])
        top_lines = top.split("; ")[:3]
        safe_score = float(row["SAFE association score"])
        risky_score = float(row["RISKY association score"])
        display_label = str(row["Display label"])
        label = (
            f"{display_label}\n"
            f"{int(row['Predicate count'])} predicates\n"
            f"Class association: SAFE {safe_score * 100:.1f}% | "
            f"RISKY {risky_score * 100:.1f}%\n"
            f"Top predicates:\n"
            + "\n".join(top_lines)
        )
        scale = 0.8 + 1.4 * int(row["Predicate count"]) / max_count
        node_name = display_label
        lines.append(
            f"  {dot_quote(node_name)} [label={dot_quote(label)}, "
            f"fillcolor={dot_quote(fill)}, color={dot_quote(border)}, "
            f"width={scale:.2f}, height={scale * 0.55:.2f}];"
        )
        safe_weight = float(row["SAFE edge weight"])
        risky_weight = float(row["RISKY edge weight"])
        if safe_weight > 0:
            pen = max(1.0, min(6.0, 1.0 + safe_weight / 50.0))
            safe_edge_label = f"{safe_score * 100:.1f}% (w={safe_weight:.0f})"
            lines.append(
                f"  {dot_quote(node_name)} -> \"Class SAFE\" "
                f"[label={dot_quote(safe_edge_label)}, penwidth={pen:.2f}];"
            )
        if risky_weight > 0:
            pen = max(1.0, min(6.0, 1.0 + risky_weight / 50.0))
            risky_edge_label = f"{risky_score * 100:.1f}% (w={risky_weight:.0f})"
            lines.append(
                f"  {dot_quote(node_name)} -> \"Class RISKY\" "
                f"[label={dot_quote(risky_edge_label)}, penwidth={pen:.2f}];"
            )
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_named_dot_file(
    dot_path: Path,
    output_dir: Path,
    stem: str,
    timeout_seconds: int,
) -> list[Path]:
    """
    Render a DOT source to PNG and SVG.

    Parameters
    ----------
    dot_path : Path
        DOT source path.
    output_dir : Path
        Directory receiving rendered files.
    stem : str
        Output filename stem.
    timeout_seconds : int
        Maximum seconds allowed for each Graphviz invocation.

    Returns
    -------
    list[Path]
        Successfully rendered image paths.
    """
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        print("Graphviz 'dot' command not found. Community DOT was saved only.")
        return []
    outputs: list[Path] = []
    for fmt in ("png", "svg"):
        path = output_dir / f"{stem}.{fmt}"
        try:
            subprocess.run(
                [dot_binary, f"-T{fmt}", str(dot_path), "-o", str(path)],
                check=True,
                timeout=timeout_seconds,
            )
            outputs.append(path)
        except subprocess.TimeoutExpired:
            print(f"Community {fmt.upper()} rendering timed out after {timeout_seconds} seconds.")
        except subprocess.CalledProcessError as exc:
            print(f"Community {fmt.upper()} rendering failed: {exc}")
    return outputs


def save_community_class_analysis(
    output_dir: Path,
    explanation: Any,
    mixed_margin: float,
    top_predicates: int,
    timeout_seconds: int,
) -> list[Path]:
    """
    Generate class-aware community artifacts.

    Parameters
    ----------
    output_dir : Path
        Directory receiving CSV, text, DOT, PNG, and SVG artifacts.
    explanation : Any
        In-memory DPG explanation.
    mixed_margin : float
        Maximum SAFE/RISKY score difference classified as ``MIXED``.
    top_predicates : int
        Number of representative predicates reported per metric.
    timeout_seconds : int
        Maximum seconds allowed for each Graphviz invocation.

    Returns
    -------
    list[Path]
        Written artifact paths.
    """
    print("\n=== Class-aware community analysis ===")
    node_metrics, edge_metrics = load_metric_tables(output_dir, explanation)
    membership = load_community_membership(output_dir, explanation)
    summary, predicates = build_community_class_tables(
        node_metrics,
        edge_metrics,
        membership,
        mixed_margin,
        top_predicates,
    )

    outputs: list[Path] = []
    summary_path = output_dir / "community_class_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    predicates_path = output_dir / "community_predicates.csv"
    predicates.to_csv(predicates_path, index=False)
    outputs.append(predicates_path)

    text_path = write_community_summary_text(
        output_dir / "community_class_summary.txt", summary
    )
    outputs.append(text_path)

    dot_path = write_community_dot(
        output_dir / "simplified_community_dpg.dot", summary
    )
    outputs.append(dot_path)
    outputs.extend(
        render_named_dot_file(
            dot_path,
            output_dir,
            "simplified_community_dpg",
            timeout_seconds,
        )
    )

    counts = summary["Dominant class"].value_counts().to_dict()
    print(f"Communities analyzed: {len(summary)}")
    print(f"Dominant-class counts: {counts}")
    return outputs


# ---------------------------------------------------------------------------
# Optional Rendering
# ---------------------------------------------------------------------------

def prepare_render_child() -> None:
    """
    Move the render worker into its own process group when supported.

    This lets the parent stop both the Python render worker and any Graphviz
    subprocess it starts if the render timeout is reached.
    """
    try:
        os.setsid()
    except OSError:
        return


def render_standard_graph_worker(
    explainer: Any,
    explanation: Any,
    output_dir: Path,
    queue: Any,
) -> None:
    """
    Render the standard DPG graph in a child process.

    Parameters
    ----------
    explainer : Any
        Fitted DPG explainer.
    explanation : Any
        DPG global explanation object.
    output_dir : Path
        Output directory.
    queue : Any
        Multiprocessing queue used to report the result.
    """
    prepare_render_child()

    try:
        explainer.plot(
            "dpg_forest",
            explanation=explanation,
            save_dir=str(output_dir),
            class_flag=True,
            layout_template="compact",
            show=False,
            export_pdf=True,
            label_mode="wrapped",
            readability="normal",
        )
        queue.put({
            "status": "ok",
            "paths": [
                str(path) for path in [
                    output_dir / "dpg_forest.png",
                    output_dir / "dpg_forest.pdf",
                ]
                if path.exists()
            ],
        })
    except Exception as exc:
        queue.put({"status": "error", "reason": str(exc)})


def render_community_graph_worker(
    explainer: Any,
    explanation: Any,
    output_dir: Path,
    queue: Any,
) -> None:
    """
    Render the community-colored DPG graph in a child process.

    Parameters
    ----------
    explainer : Any
        Fitted DPG explainer.
    explanation : Any
        DPG global explanation object.
    output_dir : Path
        Output directory.
    queue : Any
        Multiprocessing queue used to report the result.
    """
    prepare_render_child()

    try:
        explainer.plot_communities(
            "dpg_forest_communities",
            explanation=explanation,
            save_dir=str(output_dir),
            class_flag=True,
            layout_template="compact",
            show=False,
            export_pdf=True,
        )
        queue.put({
            "status": "ok",
            "paths": [
                str(path) for path in [
                    output_dir / "dpg_forest_communities.png",
                    output_dir / "dpg_forest_communities.pdf",
                ]
                if path.exists()
            ],
        })
    except Exception as exc:
        queue.put({"status": "error", "reason": str(exc)})


def stop_render_process(process: mp.Process) -> None:
    """
    Stop a render worker and its Graphviz subprocesses.

    Parameters
    ----------
    process : multiprocessing.Process
        Running render worker process.
    """
    if not process.is_alive():
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()

    process.join(timeout=5)

    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
        process.join(timeout=5)


def run_render_worker(
    worker: Any,
    args: tuple[Any, ...],
    label: str,
    timeout_seconds: int,
) -> list[Path]:
    """
    Run one graph render worker with a timeout.

    Parameters
    ----------
    worker : Any
        Worker function to run in a child process.
    args : tuple[Any, ...]
        Worker arguments before the result queue.
    label : str
        Human-readable render label for CLI messages.
    timeout_seconds : int
        Maximum seconds to wait for rendering.

    Returns
    -------
    list[Path]
        Rendered image paths, if rendering succeeds before timeout.
    """
    context = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    queue = context.Queue()
    process = context.Process(target=worker, args=(*args, queue))

    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        stop_render_process(process)
        print(
            f"\n{label} rendering timed out after {timeout_seconds} seconds. "
            "Skipping image output."
        )
        return []

    if queue.empty():
        if process.exitcode == 0:
            return []
        print(f"\n{label} rendering exited without producing output. Skipping image output.")
        return []

    result = queue.get()

    if result.get("status") != "ok":
        print(f"\n{label} rendering skipped.")
        print(f"Reason: {result.get('reason')}")
        return []

    return [Path(path) for path in result.get("paths", [])]


def try_render_graphs(
    explainer: Any,
    explanation: Any,
    output_dir: Path,
    render: bool,
    timeout_seconds: int,
    max_nodes: int,
    max_edges: int,
) -> list[Path]:
    """
    Optionally render DPG graph images.

    Rendering is disabled by default, guarded by graph size, and bounded by a
    timeout. CSV/text artifacts are the required outputs for this experiment.

    Parameters
    ----------
    explainer : Any
        Fitted DPG explainer.
    explanation : Any
        DPG global explanation object.
    output_dir : Path
        Output directory.
    render : bool
        Whether image rendering was explicitly requested.
    timeout_seconds : int
        Maximum seconds per render attempt.
    max_nodes : int
        Maximum graph nodes allowed for rendering.
    max_edges : int
        Maximum graph edges allowed for rendering.

    Returns
    -------
    list[Path]
        Rendered image paths, if rendering succeeds.
    """
    outputs: list[Path] = []
    node_count = explanation.graph.number_of_nodes()
    edge_count = explanation.graph.number_of_edges()

    if not render:
        print("Graph rendering disabled by default. Use --render to attempt image output.")
        return outputs

    if node_count > max_nodes or edge_count > max_edges:
        print("Graph rendering skipped: DPG graph is too large for the first experiment.")
        print(f"Graph size: {node_count} nodes, {edge_count} edges")
        print(f"Render limits: {max_nodes} nodes, {max_edges} edges")
        return outputs

    outputs.extend(
        run_render_worker(
            render_standard_graph_worker,
            (explainer, explanation, output_dir),
            "Standard DPG graph",
            timeout_seconds,
        )
    )

    if explanation.communities is None:
        return outputs

    outputs.extend(
        run_render_worker(
            render_community_graph_worker,
            (explainer, explanation, output_dir),
            "Community DPG graph",
            timeout_seconds,
        )
    )

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build command-line arguments for the DPG experiment.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Build DPG artifacts for the SpecLens-PML Random Forest candidate."
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Attempt optional Graphviz image rendering. Disabled by default.",
    )
    parser.add_argument(
        "--render-timeout",
        type=int,
        default=DEFAULT_RENDER_TIMEOUT_SECONDS,
        help="Maximum seconds per optional render attempt.",
    )
    parser.add_argument(
        "--max-render-nodes",
        type=int,
        default=DEFAULT_MAX_RENDER_NODES,
        help="Skip optional rendering when the DPG has more nodes than this.",
    )
    parser.add_argument(
        "--max-render-edges",
        type=int,
        default=DEFAULT_MAX_RENDER_EDGES,
        help="Skip optional rendering when the DPG has more edges than this.",
    )
    parser.add_argument(
        "--render-simplified-global",
        action="store_true",
        help="Render a simplified global DPG view from retained high-importance nodes.",
    )
    parser.add_argument(
        "--top-k-nodes",
        type=int,
        default=DEFAULT_SIMPLIFIED_TOP_K_NODES,
        help="Number of high-importance nodes retained in the simplified global view.",
    )
    parser.add_argument(
        "--node-metric",
        choices=sorted(SIMPLIFIED_NODE_METRICS),
        default=DEFAULT_SIMPLIFIED_NODE_METRIC,
        help="Node metric used to rank nodes for the simplified global view.",
    )
    parser.add_argument(
        "--max-edges",
        type=int,
        default=DEFAULT_SIMPLIFIED_MAX_EDGES,
        help="Maximum edges retained among selected simplified global nodes.",
    )
    parser.add_argument(
        "--render-community-summary",
        action="store_true",
        help="Generate class-aware community reports and a simplified community plot.",
    )
    parser.add_argument(
        "--community-mixed-margin",
        type=float,
        default=DEFAULT_COMMUNITY_MIXED_MARGIN,
        help="Maximum SAFE/RISKY score difference classified as MIXED.",
    )
    parser.add_argument(
        "--community-top-predicates",
        type=int,
        default=DEFAULT_COMMUNITY_TOP_PREDICATES,
        help="Number of representative predicates shown per community.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------

def main(
    render: bool = False,
    render_timeout: int = DEFAULT_RENDER_TIMEOUT_SECONDS,
    max_render_nodes: int = DEFAULT_MAX_RENDER_NODES,
    max_render_edges: int = DEFAULT_MAX_RENDER_EDGES,
    render_simplified_global: bool = False,
    top_k_nodes: int = DEFAULT_SIMPLIFIED_TOP_K_NODES,
    node_metric: str = DEFAULT_SIMPLIFIED_NODE_METRIC,
    max_edges: int = DEFAULT_SIMPLIFIED_MAX_EDGES,
    render_community_summary: bool = False,
    community_mixed_margin: float = DEFAULT_COMMUNITY_MIXED_MARGIN,
    community_top_predicates: int = DEFAULT_COMMUNITY_TOP_PREDICATES,
) -> None:
    """
    Run the SpecLens-PML DPG experiment.

    This is an offline explainability experiment over the Random Forest
    candidate model. It does not alter the active champion model.

    Parameters
    ----------
    render : bool
        Whether to attempt rendering of the complete DPG.
    render_timeout : int
        Maximum seconds allowed for each rendering operation.
    max_render_nodes : int
        Maximum number of nodes allowed for complete-graph rendering.
    max_render_edges : int
        Maximum number of edges allowed for complete-graph rendering.
    render_simplified_global : bool
        Whether to generate the simplified class-aware global graph.
    top_k_nodes : int
        Number of central predicate nodes retained in the simplified graph.
    node_metric : str
        Node metric used to rank predicates.
    max_edges : int
        Maximum number of edges retained in the simplified graph.
    render_community_summary : bool
        Whether to generate class-aware community reports and plots.
    community_mixed_margin : float
        Maximum SAFE/RISKY score difference classified as ``MIXED``.
    community_top_predicates : int
        Number of representative predicates shown per community.
    """
    print("=== SpecLens-PML DPG Experiment ===")
    print("Target model: models/forest.pkl")
    print("Training context: models/forest_training_context.pkl")
    print("Governance note: champion promotion logic is unchanged.")

    try:
        DPGExplainer = load_dpg_explainer()
        model = load_forest_model(FOREST_MODEL_PATH)
        feature_names, training_features, training_labels = load_training_context(
            FOREST_CONTEXT_PATH
        )

        print("\n=== Building DPG explanation ===")
        print(f"Training samples: {len(training_features)}")
        print(f"Features: {len(feature_names)}")

        explainer, explanation = build_dpg_explanation(
            DPGExplainer,
            model,
            feature_names,
            training_features,
            training_labels,
        )

        print("\n=== Saving DPG artifacts ===")
        saved_outputs = save_structured_artifacts(
            model,
            explanation,
            explainer,
            training_features,
            training_labels,
            OUTPUT_DIR,
        )

        print("\n=== Optional graph rendering ===")
        rendered_outputs = try_render_graphs(
            explainer,
            explanation,
            OUTPUT_DIR,
            render,
            render_timeout,
            max_render_nodes,
            max_render_edges,
        )

        if render_simplified_global:
            rendered_outputs.extend(
                save_simplified_global_rendering(
                    OUTPUT_DIR,
                    explanation,
                    top_k_nodes,
                    node_metric,
                    max_edges,
                    render_timeout,
                )
            )

        if render_community_summary:
            rendered_outputs.extend(
                save_community_class_analysis(
                    OUTPUT_DIR,
                    explanation,
                    community_mixed_margin,
                    community_top_predicates,
                    render_timeout,
                )
            )

    except ImportError:
        sys.exit(1)
    except FileNotFoundError as exc:
        print(f"\nMissing required artifact: {exc}")
        print("Run python3 demo.py first to generate the Random Forest model and context.")
        sys.exit(1)
    except ValueError as exc:
        print(f"\nInvalid DPG input: {exc}")
        sys.exit(1)
    except Exception as exc:
        print("\nDPG experiment failed.")
        print(f"Reason: {exc}")
        sys.exit(1)

    print("\n=== DPG experiment completed successfully ===")
    print(f"Outputs saved in: {OUTPUT_DIR}")
    print("\nSaved artifacts:")
    for path in saved_outputs + rendered_outputs:
        print(f"- {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(
        render=args.render,
        render_timeout=args.render_timeout,
        max_render_nodes=args.max_render_nodes,
        max_render_edges=args.max_render_edges,
        render_simplified_global=args.render_simplified_global,
        top_k_nodes=args.top_k_nodes,
        node_metric=args.node_metric,
        max_edges=args.max_edges,
        render_community_summary=args.render_community_summary,
        community_mixed_margin=args.community_mixed_margin,
        community_top_predicates=args.community_top_predicates,
    )
