"""Create an allowlisted Space upload folder; never copy datasets or credentials."""

import argparse
from pathlib import Path
import shutil

from carpet_ad.inference import load_run


def prepare(output):
    root = Path(__file__).resolve().parents[1]
    output = Path(output).resolve()
    if Path.cwd().resolve() != root or not output.is_relative_to(root / "artifacts"):
        raise ValueError("Run from the repository root and choose an output inside artifacts/")
    if output.exists():
        raise FileExistsError("Choose a new output folder; existing bundles are not overwritten")
    files = [Path(name) for name in ("Dockerfile", ".dockerignore", "README.md", "DATA_LICENSE.md",
        "requirements-lock.txt", "app/api.py", "data/manifests/carpet_split.csv",
        "reports/tables/clean_comparison.csv", "reports/tables/brightness_comparison.csv",
        "frontend/package.json", "frontend/package-lock.json", "frontend/index.html", "frontend/vite.config.js")]
    files += list(Path("frontend/src").glob("*")) + list(Path("src/carpet_ad").glob("*.py"))
    for run_id in ("lbp_seed42", "lbp_glcm_seed42", "patchcore_seed42"):
        metadata, _, model = load_run(run_id)  # Verify model/config/manifest/calibration before export.
        del model
        files += [Path("runs") / run_id / name for name in (metadata["artifacts"]["model"],
            "metadata.json", "config.yaml", "manifest.csv", "calibration.json", "calibration_scores.csv")]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("A required deployment file is missing")
    output.mkdir(parents=True)
    for path in files:
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    print(f"Prepared {len(files)} files in {output}; no upload performed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/space-bundle")
    prepare(parser.parse_args().output)
