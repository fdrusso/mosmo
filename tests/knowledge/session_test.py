"""Tests for mosmo.knowledge.kb.Session."""
from typing import Optional
from warnings import warn

from pymongo import MongoClient, timeout
from pymongo.errors import ConnectionFailure

from mosmo.knowledge.kb import Session, Dataset
from mosmo.model import KbEntry, DbXref

TEST = Dataset("test", "test", KbEntry)
TEST_CANON = Dataset("test", "canon", KbEntry)


class TestSession:
    """Tests for in-memory Session API, without underlying mongo DB."""

    def mem_session(self) -> Session:
        """Sets up an in-memory Session without an underlying mongo DB."""
        return Session(client=None, schema={"TEST": TEST})

    def db_session(self) -> Optional[Session]:
        """Sets up a Session with an underlying mongo DB and a fresh /test space."""
        client = MongoClient("mongodb://127.0.0.1:27017")
        try:
            with timeout(2):
                client.drop_database("test")
            return Session(client=client, schema={"TEST": TEST})
        except ConnectionFailure:
            return None

    def test_PutGet(self):
        """The KB caches and retrieves basic entries."""
        session = self.mem_session()
        obj1 = KbEntry("obj1", name="Test object 1")
        obj2 = KbEntry("obj2", name="Test object 2")
        session.put(TEST, obj1)
        session.put(TEST, obj2)

        assert len(session._cache[TEST]) == 2
        assert session.get(TEST, "obj1") is obj1

    def test_DerefObj(self):
        """The KB can dereference a DbXref."""
        session = self.mem_session()
        obj = KbEntry("obj", name="Test object")
        session.put(TEST, obj)
        assert session.deref(DbXref("TEST", "obj")) is obj
        assert session(DbXref("TEST", "obj")) is obj

    def test_DerefStr(self):
        """The KB can dereference an xref in string form."""
        session = self.mem_session()
        obj = KbEntry("17", name="Test object")
        session.put(TEST, obj)
        assert session.deref("TEST:17") is obj
        assert session("TEST:17") is obj

    def test_PutGetDeref_Canonical(self):
        """The KB retrieves preferentially from a canonical dataset."""
        session = self.mem_session()
        session.define_dataset("CANON", Dataset("test", "canon", KbEntry), canonical=True)

        # Two objects with the same id, but in different datasets
        x1 = KbEntry("x", name="Test object x")
        x2 = KbEntry("x", name="Canonical object x")
        session.put(TEST, x1)
        session.put(TEST_CANON, x2)

        assert session.get(TEST, "x") is x1
        assert session.get(TEST_CANON, "x") is x2
        assert session("x") is x2
        assert session("TEST:x") is x1

    def test_FindByName(self):
        """Find an object by its name."""
        session = self.db_session()
        if not session:
            warn("No available mongodb connection -- skipping test.")
            return

        obj = KbEntry("foo", name="The object to be found")
        session.put(TEST, obj)

        results = session.find(TEST, obj.name)
        assert len(results) == 1
        assert results[0] is obj

    def test_FindByAka(self):
        """Find an object by one of its AKAs."""
        session = self.db_session()
        if not session:
            warn("No available mongodb connection -- skipping test.")
            return

        obj = KbEntry("foo", name="The object to be found", aka=["Mark Twain", "Billy the Kid"])
        session.put(TEST, obj)

        results = session.find(TEST, obj.aka[1], include_aka=True)
        assert len(results) == 1
        assert results[0] is obj

    def test_FindByXref(self):
        """Find an object by one of its xrefs."""
        session = self.db_session()
        if not session:
            warn("No available mongodb connection -- skipping test.")
            return

        obj = KbEntry("foo", name="The object to be found", xrefs={DbXref("BAR", "bas")})
        session.put(TEST, obj)

        results = session.xref(TEST, "BAR:bas")
        assert len(results) == 1
        assert results[0] is obj
