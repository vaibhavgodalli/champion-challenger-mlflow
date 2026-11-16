# Beginner Viva Notes

1. **What is sentiment classification?** Assigning text a category such as positive or negative.
2. **What is Hugging Face?** A platform and ecosystem that hosts pretrained machine-learning models and tools to use them.
3. **What is a pretrained model?** A model whose parameters were learned earlier from training data and can be reused for a task.
4. **What is DistilBERT?** A smaller, distilled Transformer model designed to be more efficient than BERT.
5. **What is MLflow?** A tool for tracking experiments, logging metrics/artifacts/models, and managing registered model versions.
6. **What is a Champion?** The currently selected model used as the preferred production candidate.
7. **What is a Challenger?** A candidate evaluated to see whether it should replace the Champion.
8. **Why use the same evaluation inputs?** It makes the comparison fair because both models are tested on identical examples.
9. **What is accuracy?** Correct predictions divided by all predictions.
10. **What is precision?** Of the examples predicted positive, the fraction that truly are positive.
11. **What is recall?** Of the truly positive examples, the fraction correctly found.
12. **What is macro-F1?** The unweighted average of F1 scores across classes; it gives each class equal importance.
13. **Why record latency?** A more accurate model may be slower; inference speed is an operational trade-off.
14. **What is a model registry alias?** A movable name such as `champion` that points to a registered model version.
15. **Why does predict_from_mlflow.py not use Transformers?** The assignment requires final predictions to be loaded and executed through the MLflow model URI, not by directly calling a Hugging Face pipeline in that script.
16. **What is a limitation of this project?** The evaluation set is small and hand-curated, so measured scores may not generalize to real-world reviews.
