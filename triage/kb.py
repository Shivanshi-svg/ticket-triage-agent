"""Small TF-IDF knowledge base used by the agent's ``search_kb`` tool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Article:
    id: str
    category: str
    title: str
    body: str


@dataclass(frozen=True)
class KBResult:
    article: Article
    score: float


class KnowledgeBase:
    def __init__(self, articles: list[Article]) -> None:
        if not articles:
            raise ValueError("knowledge base needs at least one article")
        self.articles = articles
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(f"{a.title}. {a.body}" for a in articles)

    @classmethod
    def from_json(cls, path: str | Path = DATA_DIR / "kb.json") -> KnowledgeBase:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Article(**item) for item in raw])

    def search(
        self, query: str, category: str | None = None, min_score: float = 0.05
    ) -> KBResult | None:
        scores = cosine_similarity(self._vectorizer.transform([query]), self._matrix).ravel()
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        for i in ranked:
            if category and self.articles[i].category != category:
                continue
            if scores[i] < min_score:
                return None
            return KBResult(self.articles[i], round(float(scores[i]), 4))
        return None
