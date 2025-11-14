# evaluation/metrics.py
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from rouge_score import rouge_scorer

def classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro
    }

def rouge_l(reference_texts: List[str], predicted_texts: List[str]) -> Dict:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p)['rougeL'].fmeasure for r, p in zip(reference_texts, predicted_texts)]
    import numpy as np
    return {
        "rouge_l_mean": float(np.mean(scores)),
        "rouge_l_std": float(np.std(scores))
    }
