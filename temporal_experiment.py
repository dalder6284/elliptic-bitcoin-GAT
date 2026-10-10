"""Reusable temporal training and evaluation helpers for the Elliptic GAT notebook."""

from __future__ import annotations

import csv
import os
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import auc, average_precision_score, precision_recall_curve
from torch_geometric.data import Data


METRIC_NAMES = [
    "Loss",
    "Accuracy",
    "Precision 0",
    "Precision 1",
    "Recall 0",
    "Recall 1",
    "F1 0",
    "F1 1",
    "PR AUC",
    "Average Precision",
    "TP 0",
    "TP 1",
    "FP 0",
    "FP 1",
    "Labeled Nodes",
]


def set_seed(seed: int) -> None:
    """Seed every random-number generator used by this experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Choose an illicit-class threshold using validation labels only."""
    if np.unique(y_true).size < 2:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if thresholds.size == 0:
        return 0.5

    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    return float(thresholds[int(np.argmax(f1))])


def _classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(np.int64)

    tp_0 = int(np.sum((y_true == 0) & (y_pred == 0)))
    tp_1 = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp_0 = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp_1 = int(np.sum((y_true == 0) & (y_pred == 1)))

    precision_0 = _safe_ratio(tp_0, tp_0 + fp_0)
    precision_1 = _safe_ratio(tp_1, tp_1 + fp_1)
    recall_0 = _safe_ratio(tp_0, tp_0 + fp_1)
    recall_1 = _safe_ratio(tp_1, tp_1 + fp_0)
    f1_0 = _safe_ratio(2.0 * precision_0 * recall_0, precision_0 + recall_0)
    f1_1 = _safe_ratio(2.0 * precision_1 * recall_1, precision_1 + recall_1)

    if np.any(y_true == 1):
        curve_precision, curve_recall, _ = precision_recall_curve(y_true, scores)
        pr_auc = float(auc(curve_recall, curve_precision))
        average_precision = float(average_precision_score(y_true, scores))
    else:
        pr_auc = float("nan")
        average_precision = float("nan")

    return {
        "Accuracy": float(np.mean(y_pred == y_true)),
        "Precision 0": precision_0,
        "Precision 1": precision_1,
        "Recall 0": recall_0,
        "Recall 1": recall_1,
        "F1 0": f1_0,
        "F1 1": f1_1,
        "PR AUC": pr_auc,
        "Average Precision": average_precision,
        "TP 0": tp_0,
        "TP 1": tp_1,
        "FP 0": fp_0,
        "FP 1": fp_1,
        "Labeled Nodes": int(y_true.size),
    }


def _make_graph(
    timestep: int,
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    device: torch.device,
) -> Data:
    index = timestep - 1
    return Data(
        x=all_ts_features[index],
        edge_index=all_ts_edges[index],
        y=all_ts_labels[index],
    ).to(device)


@torch.no_grad()
def _collect_evaluation_records(
    model: nn.Module,
    timesteps: Iterable[int],
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    all_ts_labeled_masks: Sequence[torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []

    for timestep in timesteps:
        data = _make_graph(
            timestep,
            all_ts_features,
            all_ts_edges,
            all_ts_labels,
            device,
        )
        mask = all_ts_labeled_masks[timestep - 1].to(device)
        logits = model(data.x, data.edge_index)[mask]
        labels = data.y[mask]
        loss = loss_fn(logits, labels)

        records.append(
            {
                "Timestep": timestep,
                "Loss": float(loss.item()),
                "Labels": labels.detach().cpu().numpy(),
                "Scores": F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy(),
            }
        )

    return records


def _pool_records(
    records: Sequence[Mapping[str, Any]],
    threshold: float,
) -> dict[str, float | int]:
    labels = np.concatenate([record["Labels"] for record in records])
    scores = np.concatenate([record["Scores"] for record in records])
    node_counts = np.array([len(record["Labels"]) for record in records])
    losses = np.array([record["Loss"] for record in records])

    metrics = _classification_metrics(labels, scores, threshold)
    metrics["Loss"] = float(np.average(losses, weights=node_counts))
    return metrics


def evaluate_timesteps(
    model: nn.Module,
    timesteps: Iterable[int],
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    all_ts_labeled_masks: Sequence[torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    tune_threshold: bool = False,
) -> tuple[dict[str, float | int], float]:
    records = _collect_evaluation_records(
        model,
        timesteps,
        all_ts_features,
        all_ts_edges,
        all_ts_labels,
        all_ts_labeled_masks,
        loss_fn,
        device,
    )

    if tune_threshold:
        labels = np.concatenate([record["Labels"] for record in records])
        scores = np.concatenate([record["Scores"] for record in records])
        threshold = _best_f1_threshold(labels, scores)

    return _pool_records(records, threshold), threshold


def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    timesteps: Iterable[int],
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    all_ts_labeled_masks: Sequence[torch.Tensor],
    supervised_loss_fn: nn.Module,
    sst_loss_fn: nn.Module | None,
    device: torch.device,
) -> dict[str, float | int]:
    model.train()
    labels_for_epoch: list[np.ndarray] = []
    scores_for_epoch: list[np.ndarray] = []
    weighted_loss = 0.0
    labeled_nodes = 0

    for timestep in timesteps:
        data = _make_graph(
            timestep,
            all_ts_features,
            all_ts_edges,
            all_ts_labels,
            device,
        )
        mask = all_ts_labeled_masks[timestep - 1].to(device)

        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        if sst_loss_fn is None:
            loss = supervised_loss_fn(logits[mask], data.y[mask])
        else:
            # Only training timesteps enter this branch, so pseudolabels cannot
            # be created from validation or test nodes.
            loss = sst_loss_fn(logits, data.y, mask)
        loss.backward()
        optimizer.step()

        labels = data.y[mask]
        labeled_count = int(labels.numel())
        weighted_loss += float(loss.item()) * labeled_count
        labeled_nodes += labeled_count
        labels_for_epoch.append(labels.detach().cpu().numpy())
        scores_for_epoch.append(
            F.softmax(logits[mask], dim=1)[:, 1].detach().cpu().numpy()
        )

    metrics = _classification_metrics(
        np.concatenate(labels_for_epoch),
        np.concatenate(scores_for_epoch),
        threshold=0.5,
    )
    metrics["Loss"] = weighted_loss / labeled_nodes
    return metrics


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: config[key]
        for key in (
            "hidden_channels",
            "num_hidden_layers",
            "out_channels",
            "dropout",
            "num_heads",
        )
    }


def _prefixed_metrics(prefix: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix} {name}": metrics[name] for name in METRIC_NAMES}


def run_training_suite(
    *,
    model_class: type[nn.Module],
    experiments: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    train_timesteps: Sequence[int],
    validation_timesteps: Sequence[int],
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    all_ts_labeled_masks: Sequence[torch.Tensor],
    class_weights: torch.Tensor,
    sst_loss_factory: Callable[..., nn.Module],
    learning_rate: float = 0.001,
    max_epochs: int = 1000,
    patience: int = 100,
    min_delta: float = 1e-4,
    model_directory: str = "models",
    results_directory: str = "training_results",
) -> pd.DataFrame:
    """Train every configuration with every seed using validation-only selection."""
    os.makedirs(model_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = class_weights.to(device)
    manifest_rows: list[dict[str, Any]] = []

    print(f"Training on {device}; validation selects checkpoints and thresholds.")
    for experiment_name, config in experiments.items():
        for seed in seeds:
            set_seed(seed)
            model = model_class(
                in_channels=all_ts_features[0].shape[1],
                **_model_kwargs(config),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            supervised_loss_fn = nn.CrossEntropyLoss(weight=weights)
            sst_loss_fn = None
            if config.get("use_sst", False):
                sst_loss_fn = sst_loss_factory(
                    beta=config["beta"],
                    sp_weight=config["sp_weight"],
                    weight=weights,
                ).to(device)

            stem = f"balanced_{experiment_name}_training_{seed}"
            history_path = os.path.join(results_directory, stem + ".csv")
            checkpoint_path = os.path.join(model_directory, stem + ".pth")
            best_f1 = -1.0
            best_epoch = 0
            best_threshold = 0.5
            epochs_without_improvement = 0

            with open(history_path, mode="w", newline="") as history_file:
                writer = None
                for epoch in range(1, max_epochs + 1):
                    train_metrics = train_epoch(
                        model,
                        optimizer,
                        train_timesteps,
                        all_ts_features,
                        all_ts_edges,
                        all_ts_labels,
                        all_ts_labeled_masks,
                        supervised_loss_fn,
                        sst_loss_fn,
                        device,
                    )
                    validation_metrics, validation_threshold = evaluate_timesteps(
                        model,
                        validation_timesteps,
                        all_ts_features,
                        all_ts_edges,
                        all_ts_labels,
                        all_ts_labeled_masks,
                        supervised_loss_fn,
                        device,
                        tune_threshold=True,
                    )

                    validation_f1 = float(validation_metrics["F1 1"])
                    improved = validation_f1 > best_f1 + min_delta
                    if improved:
                        best_f1 = validation_f1
                        best_epoch = epoch
                        best_threshold = validation_threshold
                        epochs_without_improvement = 0
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "threshold": best_threshold,
                                "best_epoch": best_epoch,
                                "validation_f1": best_f1,
                                "validation_pr_auc": float(validation_metrics["PR AUC"]),
                                "experiment_name": experiment_name,
                                "seed": seed,
                            },
                            checkpoint_path,
                        )
                    else:
                        epochs_without_improvement += 1

                    row = {
                        "Epoch": epoch,
                        **_prefixed_metrics("Train", train_metrics),
                        **_prefixed_metrics("Validation", validation_metrics),
                        "Validation Threshold": validation_threshold,
                        "Best Checkpoint": improved,
                    }
                    if writer is None:
                        writer = csv.DictWriter(history_file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    history_file.flush()

                    if epoch == 1 or epoch % 10 == 0 or improved:
                        print(
                            f"{experiment_name} seed={seed} epoch={epoch} "
                            f"train_f1={train_metrics['F1 1']:.3f} "
                            f"val_f1={validation_f1:.3f} "
                            f"val_pr_auc={validation_metrics['PR AUC']:.3f}"
                        )

                    if epochs_without_improvement >= patience:
                        print(
                            f"Early stopping {experiment_name} seed={seed} at "
                            f"epoch {epoch}; best epoch was {best_epoch}."
                        )
                        break

            manifest_rows.append(
                {
                    "Experiment": experiment_name,
                    "Seed": seed,
                    "Best Epoch": best_epoch,
                    "Validation F1 1": best_f1,
                    "Validation Threshold": best_threshold,
                    "Epochs Run": epoch,
                    "Checkpoint": checkpoint_path,
                    "History": history_path,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(
        os.path.join(results_directory, "balanced_training_manifest.csv"),
        index=False,
    )
    return manifest


def _region_for_timestep(timestep: int) -> str:
    if timestep <= 41:
        return "Pre-shutdown"
    if timestep == 42:
        return "Shutdown"
    return "Post-shutdown"


def _summarize_across_seeds(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Loss",
        "Accuracy",
        "Precision 1",
        "Recall 1",
        "F1 1",
        "PR AUC",
        "Average Precision",
    ]
    rows: list[dict[str, Any]] = []
    for (experiment, region), group in seed_metrics.groupby(
        ["Experiment", "Region"], sort=False
    ):
        row: dict[str, Any] = {
            "Experiment": experiment,
            "Region": region,
            "Seeds": int(group["Seed"].nunique()),
        }
        for metric in metrics:
            row[f"{metric} Mean"] = float(group[metric].mean())
            row[f"{metric} Std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_checkpoint_suite(
    *,
    model_class: type[nn.Module],
    experiments: Mapping[str, Mapping[str, Any]],
    seeds: Sequence[int],
    test_timesteps: Sequence[int],
    test_regions: Mapping[str, Sequence[int]],
    all_ts_features: Sequence[torch.Tensor],
    all_ts_edges: Sequence[torch.Tensor],
    all_ts_labels: Sequence[torch.Tensor],
    all_ts_labeled_masks: Sequence[torch.Tensor],
    class_weights: torch.Tensor,
    model_directory: str = "models",
    results_directory: str = "testing_results",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate frozen validation-selected checkpoints without tuning on test data."""
    os.makedirs(results_directory, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    seed_rows: list[dict[str, Any]] = []
    all_timestep_rows: list[dict[str, Any]] = []

    for experiment_name, config in experiments.items():
        for seed in seeds:
            checkpoint_path = os.path.join(
                model_directory,
                f"balanced_{experiment_name}_training_{seed}.pth",
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            model = model_class(
                in_channels=all_ts_features[0].shape[1],
                **_model_kwargs(config),
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            threshold = float(checkpoint["threshold"])

            records = _collect_evaluation_records(
                model,
                test_timesteps,
                all_ts_features,
                all_ts_edges,
                all_ts_labels,
                all_ts_labeled_masks,
                loss_fn,
                device,
            )
            records_by_timestep = {record["Timestep"]: record for record in records}

            timestep_rows: list[dict[str, Any]] = []
            for record in records:
                metrics = _pool_records([record], threshold)
                timestep_rows.append(
                    {
                        "Experiment": experiment_name,
                        "Seed": seed,
                        "Timestep": record["Timestep"],
                        "Region": _region_for_timestep(record["Timestep"]),
                        "Threshold": threshold,
                        **metrics,
                    }
                )
            all_timestep_rows.extend(timestep_rows)
            pd.DataFrame(timestep_rows).to_csv(
                os.path.join(
                    results_directory,
                    f"balanced_{experiment_name}_testing_{seed}.csv",
                ),
                index=False,
            )

            for region_name, region_timesteps in test_regions.items():
                region_records = [
                    records_by_timestep[timestep]
                    for timestep in region_timesteps
                    if timestep in records_by_timestep
                ]
                metrics = _pool_records(region_records, threshold)
                seed_rows.append(
                    {
                        "Experiment": experiment_name,
                        "Seed": seed,
                        "Region": region_name,
                        "Best Epoch": int(checkpoint["best_epoch"]),
                        "Threshold": threshold,
                        **metrics,
                    }
                )

    timestep_metrics = pd.DataFrame(all_timestep_rows)
    timestep_metrics.to_csv(
        os.path.join(results_directory, "balanced_test_timestep_metrics.csv"),
        index=False,
    )

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 5))
    for experiment, group in timestep_metrics.groupby("Experiment", sort=False):
        by_timestep = group.groupby("Timestep")["F1 1"]
        mean = by_timestep.mean()
        std = by_timestep.std(ddof=1).fillna(0.0)
        axis.plot(mean.index, mean.values, marker="o", label=experiment)
        axis.fill_between(
            mean.index,
            (mean - std).clip(lower=0.0),
            (mean + std).clip(upper=1.0),
            alpha=0.15,
        )
    axis.axvline(42, color="black", linestyle="--", label="Shutdown (t=42)")
    axis.set(
        xlabel="Timestep",
        ylabel="Illicit-class F1",
        title="Test F1 by timestep (mean ± standard deviation across seeds)",
        ylim=(0.0, 1.0),
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        os.path.join(results_directory, "balanced_illicit_f1_by_timestep.png"),
        dpi=160,
    )
    plt.close(figure)

    seed_metrics = pd.DataFrame(seed_rows)
    seed_metrics.to_csv(
        os.path.join(results_directory, "balanced_test_seed_metrics.csv"),
        index=False,
    )
    summary = _summarize_across_seeds(seed_metrics)
    summary.to_csv(
        os.path.join(results_directory, "balanced_test_summary.csv"),
        index=False,
    )
    return seed_metrics, summary
