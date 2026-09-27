"""Frozen-model test evaluation. No fitting, tuning, or threshold updates."""

import csv
from datetime import datetime, timezone
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, roc_curve

from carpet_ad.inference import load_run, score_image
from carpet_ad.models import _hash, _run_path, _write_json
from carpet_ad.preprocessing import load_rgb, resize_rgb


def image_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or set(labels) != {0, 1}:
        raise ValueError("Image metrics require aligned binary labels with both classes")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    decisions = scores > threshold
    matrix = confusion_matrix(labels, decisions, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(labels, decisions, average="binary", zero_division=0)
    return dict(auroc=float(roc_auc_score(labels, scores)), precision=float(precision), recall=float(recall),
                f1=float(f1), false_positive_rate=float(matrix[0, 1] / matrix[0].sum()), confusion_matrix=matrix.tolist())


def pixel_auroc(masks, maps):
    if len(masks) != len(maps) or not masks or any(a.shape != b.shape for a, b in zip(masks, maps, strict=True)):
        raise ValueError("Pixel masks/maps must be nonempty and aligned")
    return float(roc_auc_score(np.concatenate([a.ravel() for a in masks]), np.concatenate([a.ravel() for a in maps])))


def _decode(path):
    with Image.open(path) as image:
        image.load()
        return image.copy()  # EXIF/RGB/resize remain inside the timed scoring function.


CONDITIONS = {"clean": 1.0, "brightness_0.8": 0.8, "brightness_1.2": 1.2}


def apply_condition(image, condition):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "clean":
        return image
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    pixels = np.asarray(rgb).astype(np.float32) * CONDITIONS[condition]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def robustness(run_id):
    run = _run_path(run_id)
    if not (run / "evaluation/clean/metrics.json").exists():
        raise ValueError("Complete clean evaluation before brightness evaluation")
    return [evaluate(run_id, condition) for condition in CONDITIONS if condition != "clean"]


def evaluate(run_id, condition="clean"):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    run = _run_path(run_id)
    output = run / "evaluation" / condition
    if output.exists():
        raise FileExistsError(f"Evaluation already exists: {output}; existing results are not overwritten")
    metadata, calibration, model = load_run(run_id)
    frozen_files = [metadata["artifacts"]["model"], "metadata.json", "config.yaml", "manifest.csv", "calibration.json", "calibration_scores.csv"]
    frozen_hashes = {name: _hash(run / name) for name in frozen_files}
    root = Path("data/raw/carpet").resolve()
    with (run / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate test manifest")
    mask_hashes = {}
    for row in rows:
        path = (root / row["path"]).resolve()
        if row["original_split"] != "test" or not path.is_relative_to(root) or not row["path"].startswith("test/"):
            raise ValueError("Invalid test image path")
        if row["label"] not in {"0", "1"} or _hash(path) != row["sha256"]:
            raise ValueError("Invalid test label or changed image hash")
        if row["label"] == "1":
            mask_path = (root / row["mask_path"]).resolve()
            if not row["mask_path"] or not mask_path.is_relative_to(root) or not row["mask_path"].startswith("ground_truth/"):
                raise ValueError("Missing or invalid anomaly mask path")
            mask_hashes[row["mask_path"]] = _hash(mask_path)
    device = str(model.memory_bank.device) if metadata["model"] == "patchcore" else "cpu"
    def synchronize():
        if device.startswith("cuda"):
            import torch
            torch.cuda.synchronize()
    warmup = apply_condition(_decode(root / rows[0]["path"]), condition)
    for _ in range(5):
        score_image(model, metadata["model"], warmup)
    synchronize()
    records, masks, maps = [], [], []
    for row in rows:
        image = apply_condition(_decode(root / row["path"]), condition)
        synchronize()
        start = perf_counter()
        score, anomaly_map = score_image(model, metadata["model"], image)
        synchronize()
        latency = (perf_counter() - start) * 1000
        records.append(dict(path=row["path"], label=int(row["label"]), defect_type=row["defect_type"],
                            condition=condition, raw_score=score, threshold=calibration["threshold"],
                            decision=int(score > calibration["threshold"]), latency_ms=latency))
        if anomaly_map is not None:
            mask = np.zeros((256, 256), dtype=np.uint8)
            if row["label"] == "1":
                with Image.open(root / row["mask_path"]) as source:
                    source.load()
                    oriented = ImageOps.exif_transpose(source)
                    values = np.asarray(oriented)
                    if values.ndim != 2 or not np.isin(values, [0, 1, 255]).all() or oriented.size != ImageOps.exif_transpose(image).size:
                        raise ValueError("Mask is not binary or aligned with its source image")
                    mask = (np.asarray(oriented.resize((256, 256), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
            masks.append(mask)
            maps.append(anomaly_map)
    frame = pd.DataFrame(records)
    summary = image_metrics(frame.label, frame.raw_score, calibration["threshold"])
    summary.update(run_id=run_id, model=metadata["model"], condition=condition, images=len(rows), device=device,
                   brightness_multiplier=CONDITIONS[condition],
                   brightness_transform="EXIF-oriented RGB uint8 -> float32 multiply -> clip [0,255] -> uint8 truncation, before resize; outside scoring timer" if condition != "clean" else None,
                   threshold=calibration["threshold"], fit_seconds=metadata["fit_seconds"],
                   latency_median_ms=float(frame.latency_ms.median()),
                   latency_p95_ms=float(np.quantile(frame.latency_ms, .95, method="linear")),
                   warmup_passes=5, batch_size=1, pixel_auroc=pixel_auroc(masks, maps) if maps else None,
                   pixel_method="global flattened nearest-neighbor-resized binary masks vs unnormalized raw maps; good masks are zero" if maps else None,
                   latency_scope="decoded image in RAM; EXIF/RGB/resize/features/scoring/output transfer included; file I/O/loading/plotting excluded; CUDA synchronized",
                   created_at=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                   platform=platform.platform(), cpu=platform.processor(), frozen_hashes=frozen_hashes,
                   mask_sha256=mask_hashes, manifest_sha256=metadata["manifest_sha256"],
                   packages={d.metadata["Name"]: d.version for d in distributions()},
                   source_sha256={p.name: _hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))})
    if device.startswith("cuda"):
        import torch
        summary["gpu"] = torch.cuda.get_device_name()
        summary["cuda_runtime"] = torch.version.cuda
    if frozen_hashes != {name: _hash(run / name) for name in frozen_files}:
        raise ValueError("Frozen model/config/calibration changed during evaluation")
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "predictions.csv", index=False)
    failures = frame[frame.label != frame.decision]
    failures.to_csv(output / "failures.csv", index=False)
    frame.assign(correct=frame.label == frame.decision).groupby("defect_type").agg(
        images=("path", "size"), predicted_anomalous=("decision", "sum"), correct=("correct", "sum")
    ).to_csv(output / "by_defect.csv")
    if maps:
        np.savez_compressed(output / "pixel_maps.npz", paths=frame.path.to_numpy(dtype=str), masks=np.stack(masks), maps=np.stack(maps))
    _plots(frame, summary, root, output, maps)
    summary["predictions_sha256"] = _hash(output / "predictions.csv")
    summary["pixel_maps_sha256"] = _hash(output / "pixel_maps.npz") if maps else None
    _write_json(output / "metrics.json", summary)  # Completion record is written last.
    return {key: summary[key] for key in ("run_id", "auroc", "precision", "recall", "f1", "false_positive_rate", "pixel_auroc", "latency_median_ms", "latency_p95_ms")}


def _plots(frame, summary, root, output, maps):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    fpr, tpr, thresholds = roc_curve(frame.label, frame.raw_score)
    pd.DataFrame(dict(fpr=fpr, tpr=tpr, threshold=thresholds)).to_csv(output / "roc.csv", index=False)
    axes[0].plot(fpr, tpr, label=f"AUROC {summary['auroc']:.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title=summary["model"])
    axes[0].legend()
    matrix = np.asarray(summary["confusion_matrix"])
    axes[1].imshow(matrix, cmap="Blues")
    for (y, x), count in np.ndenumerate(matrix):
        axes[1].text(x, y, str(count), ha="center", va="center", color="red")
    axes[1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["normal", "anomaly"], yticklabels=["normal", "anomaly"], xlabel="Predicted", ylabel="Actual", title="Frozen threshold")
    fig.tight_layout()
    fig.savefig(output / "roc_confusion.png", dpi=150)
    plt.close(fig)
    # Deterministic illustrations: first two false positives and first two misses.
    chosen = pd.concat([frame[(frame.label == 0) & (frame.decision == 1)].head(2), frame[(frame.label == 1) & (frame.decision == 0)].head(2)])
    if chosen.empty:
        chosen = frame.head(2)
    fig, axes = plt.subplots(len(chosen), 2 if maps else 1, squeeze=False, figsize=(7 if maps else 4, 3 * len(chosen)))
    for row_number, (index, row) in enumerate(chosen.iterrows()):
        axes[row_number, 0].imshow(resize_rgb(apply_condition(load_rgb(root / row.path), row.condition)))
        axes[row_number, 0].set_title(f"{row.path}\nactual {row.label}, predicted {row.decision}, score {row.raw_score:.3f}", fontsize=8)
        if maps:
            axes[row_number, 1].imshow(maps[index], cmap="inferno")
            axes[row_number, 1].set_title("Raw map; relative display colors", fontsize=8)
    for axis in axes.flat:
        axis.axis("off")
    fig.suptitle("MVTec AD / MVTec Software GmbH / CC BY-NC-SA 4.0", fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "examples.png", dpi=120)
    plt.close(fig)


def compare(run_ids):
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Comparison run IDs must be unique")
    summaries = [json.loads((_run_path(run) / "evaluation/clean/metrics.json").read_text(encoding="utf-8")) for run in run_ids]
    if len({s["manifest_sha256"] for s in summaries}) != 1:
        raise ValueError("Comparison requires the same frozen manifest")
    for run_id, summary in zip(run_ids, summaries, strict=True):
        run = _run_path(run_id)
        if _hash(run / "evaluation/clean/predictions.csv") != summary["predictions_sha256"]:
            raise ValueError("Prediction file changed")
        if summary["frozen_hashes"] != {name: _hash(run / name) for name in summary["frozen_hashes"]}:
            raise ValueError("Frozen run changed after evaluation")
    keys = ["run_id", "model", "images", "auroc", "precision", "recall", "f1", "false_positive_rate", "pixel_auroc", "fit_seconds", "device", "latency_median_ms", "latency_p95_ms"]
    output = Path("reports/tables")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{key: summary[key] for key in keys} for summary in summaries]).to_csv(output / "clean_comparison.csv", index=False)
    brightness = []
    for run_id, clean in zip(run_ids, summaries, strict=True):
        run = _run_path(run_id)
        clean_frame = pd.read_csv(run / "evaluation/clean/predictions.csv", float_precision="round_trip")
        for condition in CONDITIONS:
            folder = run / "evaluation" / condition
            if not (folder / "metrics.json").exists():
                continue
            summary = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            frame = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
            if (_hash(folder / "predictions.csv") != summary["predictions_sha256"]
                    or summary["frozen_hashes"] != clean["frozen_hashes"]
                    or summary["mask_sha256"] != clean["mask_sha256"]
                    or summary["threshold"] != clean["threshold"]
                    or not frame[["path", "label", "defect_type"]].equals(clean_frame[["path", "label", "defect_type"]])
                    or not (frame.condition == condition).all()
                    or not (frame.threshold == clean["threshold"]).all()
                    or not np.array_equal(frame.decision, frame.raw_score > clean["threshold"])):
                raise ValueError("Brightness comparison requires unchanged artifacts, paired IDs/labels and frozen decisions")
            measured = image_metrics(frame.label, frame.raw_score, clean["threshold"])
            entry = dict(run_id=run_id, model=summary["model"], condition=condition,
                         images=len(frame), threshold=clean["threshold"])
            for metric in ("auroc", "f1", "false_positive_rate"):
                if not np.isclose(measured[metric], summary[metric], rtol=0, atol=1e-12):
                    raise ValueError("Stored metrics differ from predictions")
                entry[metric] = measured[metric]
                entry[f"delta_{metric}"] = measured[metric] - clean[metric]
            entry["decisions_changed_from_clean"] = int((frame.decision != clean_frame.decision).sum())
            brightness.append(entry)
    pd.DataFrame(brightness).to_csv(output / "brightness_comparison.csv", index=False)
    return {"table": str(output / "clean_comparison.csv"), "brightness_table": str(output / "brightness_comparison.csv"), "runs": run_ids}
