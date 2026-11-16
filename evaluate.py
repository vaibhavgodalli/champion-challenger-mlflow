"""
Run two Hugging Face sentiment models on the same labelled evaluation CSV,
log metrics/artifacts/models to MLflow, and record a Champion decision.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
SEED = 42
MAX_LENGTH = 256
DEVICE = 0 if torch.cuda.is_available() else -1

MODELS = [
    {
        "run_name": "A_sentiment_distilbert-sst2",
        "short_name": "model_A",
        "hf_model_id": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion_candidate",
    },
    {
        "run_name": "B_sentiment_bert-sst2",
        "short_name": "model_B",
        "hf_model_id": "textattack/bert-base-uncased-SST-2",
        "role": "challenger_candidate",
    },
]

# Rule is fixed before examining results:
# Challenger must improve macro-F1 by at least 0.01 (one percentage point).
# If it does not, Model A remains Champion. This avoids replacing the current
# model for a tiny difference that may be noise on a small evaluation sample.
PRIMARY_METRIC = "macro_f1"
IMPROVEMENT_MARGIN = 0.01


class HuggingFaceSentimentModel(mlflow.pyfunc.PythonModel):
    """A pyfunc wrapper so inference can happen through MLflow only."""

    def load_context(self, context):
        with open(context.artifacts["model_config"], "r", encoding="utf-8") as f:
            config = json.load(f)
        self.hf_model_id = config["hf_model_id"]
        self.max_length = int(config.get("max_length", 256))
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.hf_model_id,
            tokenizer=self.hf_model_id,
            device=0 if torch.cuda.is_available() else -1,
        )

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].astype(str).tolist()
        else:
            texts = [str(x) for x in model_input]
        raw = self.classifier(texts, truncation=True, max_length=self.max_length)
        rows = []
        for text, result in zip(texts, raw):
            label = str(result["label"]).upper()
            # SST-2 label convention: LABEL_0 = negative, LABEL_1 = positive.
            if label == "LABEL_0":
                sentiment = "negative"
            elif label == "LABEL_1":
                sentiment = "positive"
            else:
                sentiment = "positive" if label.endswith("POSITIVE") else "negative"
            rows.append({"text": text, "prediction": sentiment, "score": float(result["score"])})
        return pd.DataFrame(rows)


def normalize_label(label: str) -> str:
    value = str(label).strip().lower()
    if value in {"positive", "pos", "label_1"}:
        return "positive"
    if value in {"negative", "neg", "label_0"}:
        return "negative"
    raise ValueError(f"Unknown sentiment label: {label}")


def run_model(model_info: dict, eval_df: pd.DataFrame):
    model_id = model_info["hf_model_id"]
    print(f"\nLoading {model_id} ...")
    classifier = pipeline(
        "sentiment-analysis",
        model=model_id,
        tokenizer=model_id,
        device=DEVICE,
    )

    texts = eval_df["text"].astype(str).tolist()
    start = time.perf_counter()
    raw_predictions = classifier(texts, truncation=True, max_length=MAX_LENGTH, batch_size=8)
    total_seconds = time.perf_counter() - start

    predictions = []
    scores = []
    for item in raw_predictions:
        label = str(item["label"]).upper()
        if label == "LABEL_0":
            predictions.append("negative")
        elif label == "LABEL_1":
            predictions.append("positive")
        elif "NEGATIVE" in label:
            predictions.append("negative")
        elif "POSITIVE" in label:
            predictions.append("positive")
        else:
            raise ValueError(f"Unexpected model label {label} from {model_id}")
        scores.append(float(item["score"]))

    references = [normalize_label(x) for x in eval_df["reference"]]
    metrics = {
        "accuracy": float(accuracy_score(references, predictions)),
        "precision": float(precision_score(references, predictions, pos_label="positive", zero_division=0)),
        "recall": float(recall_score(references, predictions, pos_label="positive", zero_division=0)),
        "macro_f1": float(f1_score(references, predictions, average="macro", zero_division=0)),
        "avg_latency_ms": float(total_seconds * 1000 / len(texts)),
        "total_inference_s": float(total_seconds),
    }

    result_df = eval_df.copy()
    result_df["prediction"] = predictions
    result_df["confidence"] = scores
    result_df["correct"] = result_df["reference"].map(normalize_label) == result_df["prediction"]

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / f"predictions_{model_info['short_name']}.csv"
    result_df.to_csv(prediction_path, index=False)

    metrics_path = OUTPUT_DIR / f"metrics_{model_info['short_name']}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(references, predictions, labels=["negative", "positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["negative", "positive"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix - {model_info['short_name']}")
    fig.tight_layout()
    cm_path = PLOT_DIR / f"confusion_matrix_{model_info['short_name']}.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    config_path = OUTPUT_DIR / f"model_config_{model_info['short_name']}.json"
    config_path.write_text(json.dumps({
        "hf_model_id": model_id,
        "max_length": MAX_LENGTH,
        "task": "sentiment-analysis",
    }, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=model_info["run_name"]) as run:
        mlflow.log_params({
            "hf_model_id": model_id,
            "task": "sentiment-classification",
            "role": model_info["role"],
            "max_length": MAX_LENGTH,
            "batch_size": 8,
            "eval_samples": len(eval_df),
            "seed": SEED,
            "device": "cuda" if DEVICE >= 0 else "cpu",
            "primary_metric": PRIMARY_METRIC,
        })
        mlflow.set_tags({
            "student_usn": os.getenv("STUDENT_USN", "UPDATE_WITH_YOUR_USN"),
            "dataset_name": "assignment_custom_movie_review_sentiment_subset",
            "transformers_version": __import__("transformers").__version__,
            "torch_version": torch.__version__,
            "mlflow_version": mlflow.__version__,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(prediction_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))

        # Log a portable MLflow pyfunc model. It reloads the HF model internally.
        mlflow.pyfunc.log_model(
            name="model",
            python_model=HuggingFaceSentimentModel(),
            artifacts={"model_config": str(config_path)},
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                f"transformers=={__import__('transformers').__version__}",
                f"torch=={torch.__version__}",
                f"pandas=={pd.__version__}",
            ],
        )
        run_id = run.info.run_id

    return metrics, result_df, run_id


def main():
    np.random.seed(SEED)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    eval_path = DATA_DIR / "eval_inputs.csv"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")
    eval_df = pd.read_csv(eval_path)
    if not {"text", "reference"}.issubset(eval_df.columns):
        raise ValueError("eval_inputs.csv must contain text and reference columns.")
    eval_df = eval_df.dropna(subset=["text", "reference"]).reset_index(drop=True)

    all_results = {}
    all_frames = {}
    run_ids = {}
    for info in MODELS:
        metrics, frame, run_id = run_model(info, eval_df)
        all_results[info["short_name"]] = metrics
        all_frames[info["short_name"]] = frame
        run_ids[info["short_name"]] = run_id
        print(f"{info['short_name']} metrics: {metrics}")

    # Summary artifacts and comparison run.
    summary = {
        "model_A": all_results["model_A"],
        "model_B": all_results["model_B"],
        "primary_metric": PRIMARY_METRIC,
        "improvement_margin": IMPROVEMENT_MARGIN,
        "model_A_run_id": run_ids["model_A"],
        "model_B_run_id": run_ids["model_B"],
    }
    delta = all_results["model_B"][PRIMARY_METRIC] - all_results["model_A"][PRIMARY_METRIC]
    winner = "model_B" if delta >= IMPROVEMENT_MARGIN else "model_A"
    summary["delta_primary_metric"] = delta
    summary["winner"] = winner

    summary_path = OUTPUT_DIR / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metric_names = ["accuracy", "precision", "recall", "macro_f1"]
    x = np.arange(len(metric_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, [all_results["model_A"][m] for m in metric_names], width, label="Model A")
    ax.bar(x + width/2, [all_results["model_B"][m] for m in metric_names], width, label="Model B")
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Champion vs Challenger - Evaluation Metrics")
    ax.legend()
    fig.tight_layout()
    comparison_path = PLOT_DIR / "model_comparison_metrics.png"
    fig.savefig(comparison_path, dpi=150)
    plt.close(fig)

    # Five examples where predictions disagree (if at least five exist).
    merged = all_frames["model_A"][["text", "reference", "prediction"]].merge(
        all_frames["model_B"][["text", "prediction"]],
        on="text", suffixes=("_A", "_B")
    )
    disagreements = merged[merged["prediction_A"] != merged["prediction_B"]].head(5).copy()
    disagreements["comment"] = "Models predicted different sentiment labels; inspect the review context."
    disagreements.to_csv(OUTPUT_DIR / "disagreement_examples.csv", index=False)

    with mlflow.start_run(run_name="compare_sentiment") as _:
        mlflow.log_params({
            "primary_metric": PRIMARY_METRIC,
            "margin": IMPROVEMENT_MARGIN,
            "model_A_run_id": run_ids["model_A"],
            "model_B_run_id": run_ids["model_B"],
        })
        mlflow.set_tag("winner", winner)
        mlflow.log_metric("delta_primary_metric", float(delta))
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(comparison_path))
        mlflow.log_artifact(str(OUTPUT_DIR / "disagreement_examples.csv"))
        mlflow.log_artifact(str(DATA_DIR / "eval_inputs.csv"))

    print("\n===== CHAMPION DECISION =====")
    print(f"Model A macro-F1: {all_results['model_A']['macro_f1']:.4f}")
    print(f"Model B macro-F1: {all_results['model_B']['macro_f1']:.4f}")
    print(f"Challenger improvement (B - A): {delta:.4f}")
    print(f"Decision: {winner} is the Champion under the pre-written rule.")
    print(f"Model run IDs: {run_ids}")
    print("\nNext run: python register_champion.py")


if __name__ == "__main__":
    main()
