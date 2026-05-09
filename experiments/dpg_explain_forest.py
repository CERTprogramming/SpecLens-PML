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


def select_simplified_global_tables(
    node_metrics: pd.DataFrame,
    edge_metrics: pd.DataFrame,
    metric_name: str,
    top_k_nodes: int,
    max_edges: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Select high-importance nodes and connecting edges for a readable DPG view.

    Parameters
    ----------
    node_metrics : pandas.DataFrame
        Full DPG node metric table.
    edge_metrics : pandas.DataFrame
        Full DPG edge metric table.
    metric_name : str
        Friendly metric key used for node ranking.
    top_k_nodes : int
        Number of nodes to retain.
    max_edges : int
        Maximum number of retained edges.

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
    ranked_nodes[metric_column] = pd.to_numeric(
        ranked_nodes[metric_column],
        errors="coerce",
    ).fillna(0.0)
    ranked_nodes = ranked_nodes.sort_values(
        [metric_column, "Label"],
        ascending=[False, True],
    )

    simplified_nodes = ranked_nodes.head(top_k_nodes).copy()
    selected_ids = set(simplified_nodes["Node"].astype(str))

    simplified_edges = edge_metrics[
        edge_metrics["Source_id"].astype(str).isin(selected_ids)
        & edge_metrics["Target_id"].astype(str).isin(selected_ids)
    ].copy()

    if "Weight" in simplified_edges.columns:
        simplified_edges["Weight"] = pd.to_numeric(
            simplified_edges["Weight"],
            errors="coerce",
        ).fillna(0.0)
        simplified_edges = simplified_edges.sort_values(
            ["Weight", "Source_id", "Target_id"],
            ascending=[False, True, True],
        )

    simplified_edges = simplified_edges.head(max_edges)

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
        "  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fontsize=10, margin=\"0.08,0.05\"];",
        "  edge [fontname=\"Helvetica\", fontsize=9, color=\"#6b7280\", arrowsize=0.7];",
        f"  labelloc=\"t\";",
        f"  label={dot_quote(f'SpecLens-PML simplified DPG: top {len(simplified_nodes)} nodes by {metric_name}')};",
        "",
    ]

    for _, row in simplified_nodes.iterrows():
        node_id = row["Node"]
        label = wrap_dot_label(row["Label"])
        metric_value = float(row.get(metric_column, 0.0))
        fillcolor = "#bfdbfe" if str(row["Label"]).startswith("Class ") else "#fecaca"
        display_label = f"{label}\n{metric_name}: {metric_value:.4f}"
        lines.append(
            "  "
            f"{dot_quote(node_id)} "
            "["
            f"label={dot_quote(display_label)}, "
            f"fillcolor={dot_quote(fillcolor)}, "
            "color=\"#374151\""
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
        print("Graphviz 'dot' command not found. Simplified DOT was saved without image rendering.")
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
        print(f"\n{label} rendering timed out after {timeout_seconds} seconds. Skipping image output.")
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
) -> None:
    """
    Run the first SpecLens-PML DPG experiment.

    This is an offline explainability experiment over the Random Forest
    candidate model. It does not alter the active champion model.
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
    )
