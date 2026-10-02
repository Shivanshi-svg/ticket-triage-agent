"""Tool-using triage agent. Every decision is recorded in an auditable trace.

The control flow is deterministic and rule-based on purpose: for support routing, being able to
explain *why* a ticket was escalated matters more than model creativity. The drafting step sits
behind a ``Drafter`` protocol so an LLM can be plugged in without touching the routing logic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from .classifier import TicketClassifier
from .kb import KBResult, KnowledgeBase

P1_PATTERN = re.compile(
    r"outage|down for everyone|production is down|data loss|lost all|leaked|breach", re.I
)
P2_PATTERN = re.compile(
    r"cannot|can't|error|fail|locked|charged twice|crash|timeout|times out", re.I
)


@dataclass(frozen=True)
class Ticket:
    subject: str
    body: str
    ticket_id: str = "T-0"

    @property
    def text(self) -> str:
        return f"{self.subject}. {self.body}"


@dataclass
class Step:
    tool: str
    summary: str


@dataclass
class TriageResult:
    ticket_id: str
    category: str
    confidence: float
    priority: str
    kb_article: str | None
    draft_reply: str
    escalate: bool
    escalation_reasons: list[str] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)


class Drafter(Protocol):
    def draft(self, ticket: Ticket, kb: KBResult | None, escalate: bool) -> str: ...


class TemplateDrafter:
    def draft(self, ticket: Ticket, kb: KBResult | None, escalate: bool) -> str:
        if escalate or kb is None:
            return (
                f'Hi,\n\nThanks for contacting us about "{ticket.subject}". '
                "We have passed your request to a specialist who will follow up shortly.\n\n"
                f"(Ticket {ticket.ticket_id})"
            )
        sentences = re.split(r"(?<=[.!?])\s+", kb.article.body)
        steps = " ".join(sentences[:2])
        return (
            f'Hi,\n\nThanks for contacting us about "{ticket.subject}". '
            f"{kb.article.title}: {steps}\n\n"
            "If this does not solve it, just reply and we will follow up. "
            f"(Ticket {ticket.ticket_id})"
        )


def estimate_priority(text: str, category: str) -> str:
    if category == "security" or P1_PATTERN.search(text):
        return "P1"
    if category == "feature_request":
        return "P3"
    return "P2" if P2_PATTERN.search(text) else "P3"


class TriageAgent:
    def __init__(
        self,
        classifier: TicketClassifier,
        kb: KnowledgeBase,
        drafter: Drafter | None = None,
        min_confidence: float = 0.55,
    ) -> None:
        self.classifier = classifier
        self.kb = kb
        self.drafter = drafter or TemplateDrafter()
        self.min_confidence = min_confidence

    def run(self, ticket: Ticket) -> TriageResult:
        trace: list[Step] = []

        prediction = self.classifier.predict(ticket.text)
        trace.append(
            Step("classify_ticket", f"{prediction.label} (confidence {prediction.confidence:.2f})")
        )

        priority = estimate_priority(ticket.text, prediction.label)
        trace.append(Step("estimate_priority", priority))

        low_confidence = prediction.confidence < self.min_confidence
        kb_result = self.kb.search(ticket.text, None if low_confidence else prediction.label)
        trace.append(
            Step(
                "search_kb",
                f"{kb_result.article.id} (score {kb_result.score})" if kb_result else "no match",
            )
        )

        reasons = []
        if prediction.label == "security":
            reasons.append("security incidents always go to a human")
        if priority == "P1":
            reasons.append("priority P1")
        if low_confidence:
            reasons.append(f"low classifier confidence ({prediction.confidence:.2f})")
        if kb_result is None:
            reasons.append("no relevant knowledge-base article")
        escalate = bool(reasons)
        decision = "escalate: " + "; ".join(reasons) if escalate else "auto-reply"
        trace.append(Step("decide_escalation", decision))

        draft = self.drafter.draft(ticket, kb_result, escalate)
        trace.append(Step("draft_reply", f"{len(draft)} characters"))

        return TriageResult(
            ticket_id=ticket.ticket_id,
            category=prediction.label,
            confidence=prediction.confidence,
            priority=priority,
            kb_article=kb_result.article.id if kb_result else None,
            draft_reply=draft,
            escalate=escalate,
            escalation_reasons=reasons,
            trace=trace,
        )
