"""FastAPI service: POST a ticket, receive routing decision, draft reply and audit trace."""

from __future__ import annotations

import os
from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__
from .agent import Ticket, TriageAgent
from .classifier import TicketClassifier, train_on_synthetic
from .kb import KnowledgeBase


class TicketIn(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)
    ticket_id: str = Field(default="T-0", max_length=64)


def create_app(agent: TriageAgent | None = None) -> FastAPI:
    app = FastAPI(title="Ticket Triage Agent", version=__version__)
    metrics: dict = {}
    if agent is None:
        model_path = os.getenv("TRIAGE_MODEL_PATH")
        if model_path and os.path.exists(model_path):
            model = TicketClassifier.load(model_path)
        else:
            model, metrics = train_on_synthetic()
        agent = TriageAgent(model, KnowledgeBase.from_json())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/model")
    def model_info() -> dict:
        return {"metrics": metrics, "min_confidence": agent.min_confidence}

    @app.post("/triage")
    def triage(body: TicketIn) -> dict:
        return asdict(agent.run(Ticket(body.subject, body.body, body.ticket_id)))

    return app


app = create_app()
