"""Load the Champion through MLflow alias and predict on new, unseen examples."""
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "sentiment-model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(tracking_uri=TRACKING_URI)
    alias_version = client.get_model_version_by_alias(MODEL_NAME, "champion")
    model_uri = f"models:/{MODEL_NAME}@champion"

    print(f"Loaded registered model: {MODEL_NAME}")
    print(f"Champion model version: {alias_version.version}")
    print(f"Hugging Face model ID: {alias_version.tags.get('hf_model_id', 'See source run parameters')}")

    # Important: this script uses MLflow only. No Hugging Face pipeline is imported or called.
    model = mlflow.pyfunc.load_model(model_uri)

    new_data = pd.read_csv(ROOT / "data" / "new_data.csv")
    predictions = model.predict(new_data[["text"]])
    output = pd.concat([new_data.reset_index(drop=True), predictions[["prediction", "score"]]], axis=1)
    output_path = ROOT / "outputs" / "new_data_predictions.csv"
    output_path.parent.mkdir(exist_ok=True)
    output.to_csv(output_path, index=False)

    print("\nPredictions on new data:")
    print(output.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
