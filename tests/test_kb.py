import pytest

from triage.kb import KnowledgeBase


def test_search_respects_category_filter():
    kb = KnowledgeBase.from_json()
    hit = kb.search("password reset email never arrives", category="account_access")
    assert hit.article.id == "kb-access-reset"
    # With a mismatching category filter the account article must never be returned.
    other = kb.search("password reset email never arrives", category="billing")
    assert other is None or other.article.category == "billing"


def test_search_returns_none_below_threshold():
    assert KnowledgeBase.from_json().search("xylophone zebra quartz") is None


def test_empty_kb_rejected():
    with pytest.raises(ValueError):
        KnowledgeBase([])
