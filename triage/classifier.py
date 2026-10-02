"""TF-IDF + logistic regression ticket classifier with calibrated-ish confidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import generate_tickets


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class TicketClassifier:
    def __init__(self, c: float = 5.0) -> None:
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
                ("clf", LogisticRegression(C=c, max_iter=1000)),
            ]
        )
        self.params = {"C": c, "ngram_range": "1-2"}

    def fit(self, texts: list[str], labels: list[str]) -> TicketClassifier:
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, text: str) -> Prediction:
        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        best = int(probs.argmax())
        return Prediction(
            str(classes[best]),
            round(float(probs[best]), 4),
            {str(c): round(float(p), 4) for c, p in zip(classes, probs, strict=True)},
        )

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        predicted = self.pipeline.predict(texts)
        report = classification_report(labels, predicted, output_dict=True, zero_division=0)
        return {
            "accuracy": round(report["accuracy"], 4),
            "macro_f1": round(float(f1_score(labels, predicted, average="macro")), 4),
            "per_class_f1": {
                k: round(v["f1-score"], 4) for k, v in report.items() if k in set(labels)
            },
            "n_test": len(labels),
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> TicketClassifier:
        obj = cls()
        obj.pipeline = joblib.load(path)
        return obj


def train_on_synthetic(
    n: int = 1500, seed: int = 0, test_size: float = 0.25
) -> tuple[TicketClassifier, dict]:
    """Train on synthetic tickets and return the model with held-out metrics."""
    tickets = generate_tickets(n, seed)
    texts, labels = [t for t, _ in tickets], [c for _, c in tickets]
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    model = TicketClassifier().fit(x_train, y_train)
    return model, model.evaluate(x_test, y_test)
