"""Tests for mosmo.model.base."""
from mosmo.model.base import Datasource, DS, KbEntry, DbXref


class TestDatasource:
    def test_Get(self):
        """An existing instance can be retrieved by id."""
        stuff = DS.define(Datasource(id="STUFF", name="Just some stuff"))
        assert DS.get(stuff.id) is stuff

    def test_CreateOnDemand(self):
        """A new instance is created on demand, and remains retrievable."""
        things = DS.get("THINGS")
        assert things is not None
        assert DS.get(things.id) is things

    def test_MemberAccess(self):
        """Datasource instances become part of the class definition itself."""
        junk = DS.define(Datasource(id="JUNK", name="Nothing useful"))
        assert DS.JUNK is junk


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
        assert DbXref(id="bar", db=DS.get("FOO")) == DbXref.from_str("FOO:bar")

    def test_FromStr_RoundTrip(self):
        xref_str = "FOO:bar"
        assert str(DbXref.from_str(xref_str)) == xref_str

    def test_FromStr_MissingDb(self):
        xref = DbXref.from_str("foo")
        assert xref.db is None
