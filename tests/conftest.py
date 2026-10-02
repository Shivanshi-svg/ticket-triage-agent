import pytest

from triage.agent import TriageAgent
from triage.classifier import train_on_synthetic
from triage.kb import KnowledgeBase


@pytest.fixture(scope="session")
def trained():
    return train_on_synthetic()


@pytest.fixture(scope="session")
def agent(trained):
    model, _ = trained
    return TriageAgent(model, KnowledgeBase.from_json())
