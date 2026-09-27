"""Command entry point; experiment commands arrive in subsequent tasks."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Carpet anomaly detection experiments.",
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("doctor", help="Check imports and CPU/CUDA tensor execution; no downloads.")
    prepare = commands.add_parser("prepare-data", help="Audit local carpet files and freeze the split manifest.")
    prepare.add_argument("--data-root", default="data/raw/carpet")
    prepare.add_argument("--seed", type=int, default=42)
    fitting = commands.add_parser("fit", help="Fit a normal-only model; never overwrite a run.")
    fitting.add_argument("--config", required=True)
    fitting.add_argument("--run-id", required=True)
    calibration = commands.add_parser("calibrate", help="Calibrate a model on held-out normal images.")
    calibration.add_argument("--run-id", required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate a frozen model on the official test set.")
    evaluation.add_argument("--run-id", required=True)
    evaluation.add_argument("--condition", choices=["clean", "brightness_0.8", "brightness_1.2"], default="clean")
    brightness = commands.add_parser("robustness", help="Evaluate paired brightness conditions using frozen thresholds.")
    brightness.add_argument("--run-id", required=True)
    comparison = commands.add_parser("compare", help="Compare measured clean results sharing one manifest.")
    comparison.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    if args.command in {"evaluate", "compare", "robustness"}:
        from carpet_ad.evaluation import evaluate, compare, robustness
        try:
            if args.command == "robustness":
                result = robustness(args.run_id)
            else:
                result = evaluate(args.run_id, args.condition) if args.command == "evaluate" else compare(args.runs)
        except (OSError, ValueError) as error:
            print(str(error))
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command in {"fit", "calibrate"}:
        from carpet_ad.models import fit, calibrate

        try:
            result = fit(args.config, args.run_id) if args.command == "fit" else calibrate(args.run_id)
        except (OSError, ValueError) as error:
            print(str(error))
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "prepare-data":
        from carpet_ad.data import prepare_data

        try:
            result = prepare_data(args.data_root, "data/manifests/carpet_split.csv", "reports/data_audit.md", args.seed)
        except (OSError, ValueError) as error:
            print(str(error))
            return 1
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "doctor":
        from carpet_ad.doctor import diagnose

        report = diagnose()
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
