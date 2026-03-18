"""Tests for async probe safety in ElenchusProbe."""
from src.elenchus import ElenchusProbe


def test_semaphore_not_created_in_init():
    """Semaphore must not be created in __init__ — no running event loop there."""
    probe = ElenchusProbe(model="test")
    assert probe.semaphore is None
