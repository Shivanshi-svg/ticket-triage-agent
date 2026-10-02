"""Train the classifier on synthetic tickets, print metrics and optionally log to MLflow.

Usage: python -m triage.train [--out artifacts/model.joblib] [--mlflow]
"""

from __future__ import annotations

import argparse
import json

from .classifier import train_on_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/model.joblib")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--mlflow", action="store_true", help="log params/metrics to MLflow")
    args = parser.parse_args()

    model, metrics = train_on_synthetic(args.n)
    model.save(args.out)
    print(json.dumps(metrics, indent=2))

    if args.mlflow:
        import mlflow

        with mlflow.start_run(run_name="tfidf-logreg"):
            mlflow.log_params({**model.params, "n_tickets": args.n})
            mlflow.log_metrics({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
            mlflow.log_artifact(args.out)


if __name__ == "__main__":
    main()
