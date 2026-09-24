#!/usr/bin/env python
"""
This step takes the best model, tagged with the "prod" alias,
and tests it against the test dataset.
"""
import argparse
import logging
from pathlib import Path

import mlflow
import pandas as pd
import wandb
from sklearn.metrics import mean_absolute_error


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def artifact_root(reference: str) -> str:
    """Convert a W&B artifact reference into a Windows-safe local path."""
    safe_name = (
        reference
        .replace("/", "__")
        .replace("\\", "__")
        .replace(":", "__")
    )
    return str(Path("artifacts") / safe_name)


def go(args):
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading model artifact")
    model_artifact = run.use_artifact(args.mlflow_model)
    model_local_path = model_artifact.download(
        root=artifact_root(args.mlflow_model)
    )

    logger.info("Downloading test dataset artifact")
    test_artifact = run.use_artifact(args.test_dataset)
    test_dataset_path = test_artifact.file(
        root=artifact_root(args.test_dataset)
    )

    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test, y_test)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info("Score: %s", r_squared)
    logger.info("MAE: %s", mae)

    run.summary["r2"] = r_squared
    run.summary["mae"] = mae

    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model against the test dataset"
    )

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLflow model",
        required=True,
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True,
    )

    args = parser.parse_args()
    go(args)