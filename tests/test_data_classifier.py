from triage.classifier import TicketClassifier
from triage.data import CATEGORIES, generate_tickets


def test_generator_is_deterministic_and_balanced():
    a, b = generate_tickets(100, seed=3), generate_tickets(100, seed=3)
    assert a == b
    labels = [c for _, c in a]
    assert all(labels.count(c) == 20 for c in CATEGORIES)


def test_held_out_metrics(trained):
    _, metrics = trained
    assert metrics["macro_f1"] >= 0.9
    assert all(f1 >= 0.8 for f1 in metrics["per_class_f1"].values())


def test_prediction_probabilities_sum_to_one(trained):
    model, _ = trained
    p = model.predict("I cannot log in, the password reset email never arrives")
    assert p.label == "account_access"
    assert abs(sum(p.probabilities.values()) - 1) < 0.01


def test_save_and_load_roundtrip(trained, tmp_path):
    model, _ = trained
    path = tmp_path / "m.joblib"
    model.save(path)
    text = "Please refund the duplicate payment"
    assert TicketClassifier.load(path).predict(text).label == model.predict(text).label
