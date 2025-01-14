"""Tests for mosmo.knowledge.kb.Session."""
from typing import Optional
from warnings import warn
import pytest

from pymongo import MongoClient, timeout
from pymongo.errors import ConnectionFailure

from mosmo.knowledge.codecs import CODECS
from mosmo.knowledge.session import Session, Dataset
from mosmo.model import KbEntry, DbXref, DS

TEST = Dataset("TEST", DS.get("TEST"), KbEntry, "test", "test", codec=CODECS[KbEntry])
TEST_CANON = Dataset("CANON", DS.get("CANON"), KbEntry, "test", "canon", codec=CODECS[KbEntry], canonical=True)


class TestSession:
    """Tests for Session API, with or without underlying mongo DB."""

    def mem_session(self) -> Session:
        """Sets up an in-memory Session without an underlying mongo DB."""
        return Session(client=None, schema=[TEST])

    def db_session(self) -> Optional[Session]:
        """Sets up a Session with an underlying mongo DB and a fresh /test space."""
        client = MongoClient()
        try:
            with timeout(2):
                client.drop_database(TEST.client_db)
            return Session(client=client, schema=[TEST])
        except ConnectionFailure:
            return None

    def test_DatasetIsMember(self):
        """The session provides access to its schema via instance members."""
        session = self.mem_session()
        assert session.schema["TEST"] is TEST
        assert session.TEST is TEST

    def test_WriteLock(self):
        """The KB does not allow writing when locked."""
        session = self.mem_session()
        obj1 = KbEntry("obj1", name="Test object 1")
        with pytest.raises(ValueError):
            session.put(TEST, obj1)

    def test_PutGet(self):
        """The KB caches and retrieves basic entries."""
        session = self.mem_session()
        obj1 = KbEntry("obj1", name="Test object 1")
        obj2 = KbEntry("obj2", name="Test object 2")
        with session.unlock(TEST):
            session.put(TEST, obj1)
            session.put(TEST, obj2)

        assert len(session._cache[TEST]) == 2
        assert session.get(TEST, "obj1") is obj1

    def test_DerefObj(self):
        """The KB can dereference a DbXref."""
        session = self.mem_session()
        obj = KbEntry("obj", name="Test object")
        with session.unlock(TEST):
            session.put(TEST, obj)

        assert session.deref(obj.ref()) is obj

    def test_DerefStr(self):
        """The KB can dereference an xref in string form."""
        session = self.mem_session()
        obj = KbEntry("17", name="Test object")
        with session.unlock(TEST):
            session.put(TEST, obj)

        assert session.deref(str(obj.ref())) is obj

    def test_ShortcutAccess(self):
        """Tests the KB's "shortcut" access pattern."""
        session = self.mem_session()
        obj1 = KbEntry("obj1", name="Test object 1")
        obj2 = KbEntry("obj2", name="Test object 2")
        with session.unlock(TEST):
            session.put(TEST, obj1)
            session.put(TEST, obj2)

        assert session("obj1") is obj1
        assert session("TEST:obj2") is obj2
        assert session("never heard of him") is None

    def test_PutGetDeref_Canonical(self):
        """The KB retrieves preferentially from a canonical dataset."""
        session = self.mem_session()
        session.define_dataset(TEST_CANON)

        # Two objects with the same id, but in different datasets
        x1 = KbEntry("x", name="Test object x")
        x2 = KbEntry("x", name="Canonical object x")
        with session.unlock(TEST, TEST_CANON):
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
        with session.unlock(TEST):
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
        with session.unlock(TEST):
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

        obj = KbEntry("foo", name="The object to be found", xrefs={DbXref.from_str("BAR:bas")})
        with session.unlock(TEST):
            session.put(TEST, obj)

        results = session.xref(TEST, "BAR:bas")
        assert len(results) == 1
        assert results[0] is obj

    def test_UpdateDb(self):
        """An entry's db attribute is updated to reflect where it is persisted."""
        session = self.mem_session()
        obj = KbEntry("obj", name="The object.")
        assert obj.db is None
        with session.unlock(TEST):
            session.put(TEST, obj)
        assert obj.db == TEST.datasource

    def test_PutCopy(self):
        """Persisting an entry to a new dataset makes a copy."""
        session = self.mem_session()
        session.define_dataset(Dataset("METOO", DS.get("METOO"), KbEntry, "test", "metoo", codec=CODECS[KbEntry]))
        obj = KbEntry("obj", name="The object.")
        with session.unlock(TEST, session.METOO):
            session.put(TEST, obj)
            session.put(session.METOO, obj)
        clone = session("METOO:obj")

        assert session("TEST:obj") is obj
        assert clone is not None
        assert clone.db == session.METOO.datasource
        assert clone != obj
