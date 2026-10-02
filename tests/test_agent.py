import pytest

from triage.agent import Ticket, estimate_priority


def test_routine_ticket_is_auto_replied_with_kb_article(agent):
    r = agent.run(Ticket("Login help needed", "The password reset email never arrives.", "T-1"))
    assert r.category == "account_access" and not r.escalate
    assert r.kb_article == "kb-access-reset"
    assert "T-1" in r.draft_reply and "Forgot password" in r.draft_reply


def test_security_ticket_always_escalates_as_p1(agent):
    ticket = Ticket("Suspicious activity", "Someone accessed my account without permission.")
    r = agent.run(ticket)
    assert r.category == "security" and r.priority == "P1" and r.escalate
    assert any("security" in reason for reason in r.escalation_reasons)
    assert "specialist" in r.draft_reply


def test_gibberish_escalates_for_low_confidence(agent):
    r = agent.run(Ticket("asdf", "qwer zxcv uiop"))
    assert r.escalate
    assert any("low classifier confidence" in reason for reason in r.escalation_reasons)


def test_trace_records_every_tool_in_order(agent):
    r = agent.run(Ticket("Billing question", "I was charged twice for my Nimbus subscription."))
    assert [s.tool for s in r.trace] == [
        "classify_ticket", "estimate_priority", "search_kb", "decide_escalation", "draft_reply",
    ]


@pytest.mark.parametrize(
    "text,category,expected",
    [
        ("production is down for everyone", "technical", "P1"),
        ("The app crashes on upload", "technical", "P2"),
        ("Please add dark mode, I cannot find it", "feature_request", "P3"),
        ("Where can I find my invoice", "billing", "P3"),
    ],
)
def test_priority_rules(text, category, expected):
    assert estimate_priority(text, category) == expected
