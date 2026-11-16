# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and models

Task: Binary sentiment classification of movie-review text.

Evaluation data: `data/eval_inputs.csv` (50 labelled examples). Both models receive the same examples in the same order.

- Model A: `distilbert/distilbert-base-uncased-finetuned-sst-2-english`
- Model B: `textattack/bert-base-uncased-SST-2`

New examples for the final inference step are in `data/new_data.csv` and are separate from the evaluation set.

## 2. How to run (Windows PowerShell)

Open this folder in VS Code. Use Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Start the MLflow tracking server in Terminal 1:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python -m mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "$PWD/mlartifacts"
```

Keep that terminal open. In Terminal 2, activate the same environment, then run:

```powershell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
```

Open the MLflow UI at http://127.0.0.1:5000.

The first evaluation run downloads pretrained models and can take time. A working internet connection is required.

## 3. Champion rule (written before the experiment)

Primary metric: macro-F1 (higher is better). The Challenger (Model B) replaces Model A only if its macro-F1 is at least 0.01 higher. Otherwise Model A remains Champion. The one-percentage-point margin avoids replacing the baseline for a tiny observed difference on a small sample. Latency is reported as a secondary trade-off; it does not override the pre-declared primary-metric rule.

## 4. Results

The evaluation was performed on the same 50 sentiment examples for both models.

| Metric | Model A | Model B |
|---|---:|---:|
| Accuracy | 0.96 | 0.98 |
| Macro-F1 | 0.9589 | 0.9793 |
| Average latency (ms) | 12.09 | 18.67 |

Model B achieved better classification performance, while Model A had lower average inference latency.

## 5. Decision

Model B was selected as the champion based on the predefined Macro-F1 rule.

The comparison run recorded Model B as the winner, with a Macro-F1 improvement of approximately 0.0204 over Model A. This exceeded the required improvement threshold of 0.01.

Therefore:
- Champion: Model B
- Registered model: sentiment-model
- Champion version: 2
- Hugging Face model: textattack/bert-base-uncased-SST-2
- Challenger: Model A

## 6. Limitations and declaration

- This is a small, hand-curated teaching evaluation set, not a representative benchmark.
- Results may change with different examples, wording, or model/library versions.
- The models are pretrained sentiment classifiers; they are not trained from scratch in this project.
- AI assistant declaration: ChatGPT was used to help understand the assignment, scaffold code, and explain implementation. The student must run, inspect, understand, and verify the final work.

## Required submission contents

- `evaluate.py`
- `register_champion.py`
- `predict_from_mlflow.py`
- `requirements.txt`
- `data/eval_inputs.csv` and `data/new_data.csv`
- `outputs/` with generated prediction files, metrics and plots
- `screenshots/` with: MLflow comparison view, model registry aliases, and terminal output of `predict_from_mlflow.py`

Do not commit `.venv/`, `mlruns/`, `mlflow.db`, or `mlartifacts/`. Commit the project at least three times during development.
