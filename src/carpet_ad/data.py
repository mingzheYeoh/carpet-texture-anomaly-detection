"""Audit original MVTec AD carpet files and freeze a normal-only split."""

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def prepare_data(data_root, manifest, audit, seed=42):
    root, manifest, audit = Path(data_root).resolve(), Path(manifest), Path(audit)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    issues, rows = [], []
    if not root.is_dir():
        issues.append("Dataset directory is missing; download and extract original MVTec AD carpet first")
    dimensions, channels, mask_dimensions = Counter(), Counter(), Counter()
    image_files = sorted((p for folder in ("train", "test") for p in (root / folder).rglob("*") if p.is_file()),
                         key=lambda p: p.relative_to(root).as_posix())
    mask_files = sorted(p for p in (root / "ground_truth").rglob("*") if p.is_file())
    masks = defaultdict(list)
    for path in mask_files:
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or not path.stem.endswith("_mask"):
            issues.append(f"Unexpected mask path: {relative.as_posix()}")
        else:
            masks[(relative.parts[1], path.stem[:-5])].append(path)
    used_masks = set()
    for path in image_files:
        relative = path.relative_to(root)
        if len(relative.parts) != 3 or (relative.parts[0] == "train" and relative.parts[1] != "good"):
            issues.append(f"Unexpected image path: {relative.as_posix()}")
            continue
        original_split, defect_type, _ = relative.parts
        label = int(defect_type != "good")
        try:
            if not path.resolve().is_relative_to(root):
                raise ValueError("Image path escapes data root")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source)
                size = image.size
                dimensions[str(size)] += 1
                channels[f"{image.mode} ({len(image.getbands())} channels)"] += 1
            mask_path = ""
            if label:
                matches = masks[(defect_type, path.stem)]
                if len(matches) != 1:
                    raise ValueError("Missing or ambiguous mask")
                mask_file = matches[0]
                if not mask_file.resolve().is_relative_to(root):
                    raise ValueError("Mask path escapes data root")
                if mask_file in used_masks:
                    raise ValueError("Mask is shared by multiple images")
                used_masks.add(mask_file)
                with Image.open(mask_file) as source:
                    source.load()
                    mask = ImageOps.exif_transpose(source)
                    mask_dimensions[str(mask.size)] += 1
                    values = np.asarray(mask)
                    if mask.size != size:
                        raise ValueError(f"Mask dimensions {mask.size} differ from image {size}")
                    if values.ndim != 2 or not np.isin(values, [0, 1, 255]).all() or not values.any():
                        raise ValueError("Mask must be a nonempty binary single-channel image")
                mask_path = mask_file.relative_to(root).as_posix()
            rows.append(dict(path=relative.as_posix(), sha256=digest,
                             original_split=original_split, split="test", label=label,
                             defect_type=defect_type, mask_path=mask_path))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            issues.append(f"{relative.as_posix()}: {error}")
    for path in sorted(set(mask_files) - used_masks):
        issues.append(f"Orphan mask: {path.relative_to(root).as_posix()}")

    fitting_pool = [row for row in rows if row["original_split"] == "train"]
    if len(fitting_pool) < 2:
        issues.append("At least two readable train/good images are required")
    if not any(row["original_split"] == "test" and row["label"] == 0 for row in rows):
        issues.append("No readable test/good images")
    if not any(row["original_split"] == "test" and row["label"] == 1 for row in rows):
        issues.append("No readable anomalous test images with masks")
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.permutation(len(fitting_pool))
    boundary = int(np.floor(0.8 * len(fitting_pool)))
    for position, index in enumerate(order):
        fitting_pool[index]["split"] = "fit" if position < boundary else "calibration"
    hashes = defaultdict(list)
    for row in rows:
        hashes[row["sha256"]].append(row)
    duplicates = [group for group in hashes.values() if len(group) > 1]
    for group in duplicates:
        if len({row["split"] for row in group}) > 1:
            issues.append("Cross-split duplicate: " + ", ".join(row["path"] for row in group))

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=["path", "sha256", "original_split", "split", "label", "defect_type", "mask_path"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    content = output.getvalue()
    if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
        issues.append("Refusing to change an existing manifest; preserve it and investigate the source/seed change")
    counts = Counter((row["original_split"], row["defect_type"]) for row in rows)
    snapshot = "\n".join([
        f"Status: {'FAILED' if issues else 'PASSED'}.",
        f"Data root: `{root.name}` (paths relative to category root).",
        f"Seed: {seed}; NumPy {np.__version__}; Generator(PCG64); sorted POSIX paths; floor(0.8*N) fit.",
        f"Discovered images: {len(image_files)}; masks: {len(mask_files)}; validated image/mask pairs or normal images: {len(rows)}.",
        f"Image dimensions after EXIF orientation: {dict(dimensions)}.",
        f"Image modes: {dict(channels)}; mask dimensions: {dict(mask_dimensions)}.",
        f"Assigned split counts: {dict(Counter(row['split'] for row in rows))}.",
        f"Duplicate image SHA-256 groups: {len(duplicates)} (byte-identical files only; no perceptual duplicate claim).",
        "", "| Original split | Type | Validated count |", "| --- | --- | --- |",
        *[f"| {split} | {kind} | {count} |" for (split, kind), count in sorted(counts.items())],
        "", "Duplicate groups:",
        *["- " + ", ".join(row["path"] for row in group) for group in duplicates],
        "", "Issues:", *([f"- {issue}" for issue in issues] or ["- None."]),
        "", "No image pixels, labels, masks, or thresholds were modified. No model fitting was performed.", "",
    ])
    audit.parent.mkdir(parents=True, exist_ok=True)
    if not audit.exists() or not audit.read_text(encoding="utf-8").endswith(snapshot):
        with audit.open("a", encoding="utf-8") as stream:
            stream.write(f"\n# Data audit — {datetime.now(timezone.utc).isoformat()}\n\n" + snapshot)
    if issues:
        raise ValueError(f"Data audit failed; see {audit}: " + "; ".join(issues))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    if not manifest.exists():
        with manifest.open("x", encoding="utf-8", newline="") as stream:
            stream.write(content)
    return {"images": len(rows), "splits": dict(Counter(row["split"] for row in rows)),
            "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
