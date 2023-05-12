"""Tests for mosmo.model.base."""
from mosmo.model.base import KbEntry, DbXref


class TestKbEntry:
    def test_ValueSemantics(self):
        """KbEntry objects follow value semantics."""
        a = KbEntry("a")
        _a = KbEntry("a")
        assert a is not _a
        assert a == _a

    def test_Hashable(self):
        """KbEntry is hashable, and can be used as a dict key."""
        a = KbEntry("a")
        b = KbEntry("b")
        counts = {a: 3, b: 1}
        assert counts[a] == 3
        assert counts[b] == 1


class TestDbXref:
    def test_FromStr(self):
        assert DbXref("FOO", "bar") == DbXref.from_str("FOO:bar")

    def test_FromStr_RoundTrip(self):
        xref_str = "FOO:bar"
        assert str(DbXref.from_str(xref_str)) == xref_str

    def test_FromStr_MissingDb(self):
        xref = DbXref.from_str("foo")
        assert xref.db is None
