"""
CAPA-Fi: Evaluation & Benchmarking Package (Member 4 - Fathima Bushara M. C.)

Exposes evaluation metrics, partial-shift benchmarking, and visualization routines:
    - match_slots_hungarian: Hungarian bipartite matching for slot evaluation
    - compute_slot_wise_accuracy: Slot-level classification accuracy
    - compute_occupancy_mae: Mean Absolute Error on concurrent user count
    - compute_macro_f1: Macro-averaged F1 score
    - compute_exact_match_ratio: Strict multi-slot exact match accuracy
    - evaluate_predictions: Complete evaluation dictionary
    - plot_confusion_matrix: Activity confusion matrix visualizer
    - plot_tsne_embeddings: 2D t-SNE latent feature manifold visualizer
    - plot_loss_curves: Target adaptation loss telemetry curves
    - export_latex_table: Automated LaTeX table exporter
    - run_benchmark: End-to-end benchmark evaluation runner
"""

from src.evaluation.metrics import (
    match_slots_hungarian,
    compute_slot_wise_accuracy,
    compute_occupancy_mae,
    compute_macro_f1,
    compute_exact_match_ratio,
    evaluate_predictions,
)
from src.evaluation.benchmark import (
    plot_confusion_matrix,
    plot_tsne_embeddings,
    plot_loss_curves,
    export_latex_table,
    run_benchmark,
)

__all__ = [
    "match_slots_hungarian",
    "compute_slot_wise_accuracy",
    "compute_occupancy_mae",
    "compute_macro_f1",
    "compute_exact_match_ratio",
    "evaluate_predictions",
    "plot_confusion_matrix",
    "plot_tsne_embeddings",
    "plot_loss_curves",
    "export_latex_table",
    "run_benchmark",
]
