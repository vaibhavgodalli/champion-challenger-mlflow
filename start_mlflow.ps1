# Run this from the project root in an activated .venv terminal.
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "$PWD/mlartifacts"
