"""Register both logged MLflow models and assign champion/challenger aliases."""
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
MODEL_NAME = "sentiment-model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("Experiment not found. Run evaluate.py first.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        max_results=100,
        order_by=["attributes.start_time DESC"],
    )
    model_runs = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        if name.startswith("A_sentiment_") and "model_A" not in model_runs:
            model_runs["model_A"] = run.info.run_id
        elif name.startswith("B_sentiment_") and "model_B" not in model_runs:
            model_runs["model_B"] = run.info.run_id

    summary_path = ROOT / "outputs" / "metrics_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError("outputs/metrics_summary.json missing. Run evaluate.py first.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    # Use run IDs recorded by evaluate.py, avoiding accidental selection of older runs.
    model_runs = {
        "model_A": summary["model_A_run_id"],
        "model_B": summary["model_B_run_id"],
    }
    winner = summary["winner"]
    loser = "model_B" if winner == "model_A" else "model_A"

    registered = {}
    for key in ("model_A", "model_B"):
        run_id = model_runs[key]
        source_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri=source_uri, name=MODEL_NAME)
        registered[key] = result.version
        client.set_model_version_tag(MODEL_NAME, result.version, "hf_model_id",
            "distilbert/distilbert-base-uncased-finetuned-sst-2-english" if key == "model_A"
            else "textattack/bert-base-uncased-SST-2")

    client.set_registered_model_alias(MODEL_NAME, "champion", str(registered[winner]))
    client.set_registered_model_alias(MODEL_NAME, "challenger", str(registered[loser]))

    print(f"Registered model: {MODEL_NAME}")
    print(f"Model A version: {registered['model_A']}")
    print(f"Model B version: {registered['model_B']}")
    print(f"Champion alias -> version {registered[winner]} ({winner})")
    print(f"Challenger alias -> version {registered[loser]} ({loser})")
    print("\nNext run: python predict_from_mlflow.py")


if __name__ == "__main__":
    main()
