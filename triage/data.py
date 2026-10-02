"""Deterministic synthetic ticket generator (no real customer data)."""

from __future__ import annotations

import random

PRODUCTS = ["Nimbus", "Orbit", "Pulse", "Atlas"]

SENTENCES: dict[str, list[str]] = {
    "billing": [
        "I was charged twice for my {product} subscription.",
        "Please refund the duplicate payment on my invoice.",
        "My invoice shows the wrong amount for this month.",
        "I need to update my credit card for billing.",
        "Why did the price of my {product} plan increase?",
        "The VAT number is missing from my receipt.",
    ],
    "technical": [
        "The {product} app crashes when I upload a file.",
        "I get a 500 error when calling the API.",
        "The dashboard is very slow and times out.",
        "Sync with {product} fails with a timeout error.",
        "Export to CSV returns an empty file.",
        "The integration stopped working after the latest update.",
    ],
    "account_access": [
        "I cannot log in to my {product} account.",
        "The password reset email never arrives.",
        "My account is locked after too many attempts.",
        "Two-factor codes are not accepted.",
        "I need to change the email address on my account.",
        "I was removed from the workspace and lost access.",
    ],
    "feature_request": [
        "It would be great if {product} supported dark mode.",
        "Please add an option to export reports as PDF.",
        "We would like single sign-on support.",
        "Can you add a bulk edit feature?",
        "A Slack integration would help our team.",
        "Is there a roadmap for a mobile widget?",
    ],
    "security": [
        "I received a phishing email pretending to be {product}.",
        "I think someone accessed my account without permission.",
        "We found an exposed API key in a public repository.",
        "There are suspicious logins from another country.",
        "I want to report a possible vulnerability in {product}.",
        "Our data may have been leaked.",
    ],
}

SUBJECTS: dict[str, list[str]] = {
    "billing": ["Billing question", "Payment problem", "Invoice issue"],
    "technical": ["Something is broken", "Technical issue", "Error in the app"],
    "account_access": ["Access problem", "Login help needed", "Account question"],
    "feature_request": ["Feature idea", "Suggestion", "Product feedback"],
    "security": ["Urgent: security concern", "Suspicious activity", "Possible incident"],
}

CATEGORIES = sorted(SENTENCES)
GREETINGS = ["Hi team,", "Hello,", "Good morning,", ""]


def generate_tickets(n: int = 1500, seed: int = 0, noise: float = 0.15) -> list[tuple[str, str]]:
    """Return ``[(text, category)]``. ``noise`` adds an off-topic sentence to some tickets."""
    rng = random.Random(seed)
    tickets = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        product = rng.choice(PRODUCTS)
        parts = rng.sample(SENTENCES[category], k=2)
        if rng.random() < noise:
            other = rng.choice([c for c in CATEGORIES if c != category])
            parts.insert(rng.randrange(len(parts) + 1), rng.choice(SENTENCES[other]))
        body = " ".join(parts).format(product=product)
        subject = rng.choice(SUBJECTS[category])
        text = f"{subject}. {rng.choice(GREETINGS)} {body}".replace("  ", " ").strip()
        tickets.append((text, category))
    return tickets
