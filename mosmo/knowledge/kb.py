"""Knowledge Base for Molecular Systems Modeling."""
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Type

import pymongo

from mosmo.knowledge import codecs
from mosmo.model.base import DbXref, KbEntry
from mosmo.model.core import Molecule, Reaction, Pathway


@dataclass(eq=True, order=True, frozen=True)
class Dataset:
    db: str
    collection: str
    content_type: Type[KbEntry]
    codec: codecs.Codec = None

    def __repr__(self):
        return f'{self.db}.{self.collection} [{self.content_type.__name__}]'


class Session:
    def __init__(self, uri: str = 'mongodb://127.0.0.1:27017', schema: Mapping[str, Dataset] = None):
        self._uri = uri
        self._client: Optional[pymongo.MongoClient] = None
        self.schema: Dict[str, Dataset] = {}
        self.canon: Dict[Type[KbEntry], Dataset] = {}
        self._cache: Dict[Dataset, Dict[Any, KbEntry]] = {}

        if schema:
            for name, dataset in schema.items():
                self.define_dataset(name, dataset)

    def define_dataset(self, name: str, dataset: Dataset, canonical=False):
        """Adds a dataset to the schema of this session."""
        if name not in self.schema and name not in self.__dict__:
            self.schema[name] = dataset
            if canonical:
                if dataset.content_type in self.canon:
                    raise ValueError(f"Attempt to redefine canonical dataset for {dataset.content_type}.")
                self.canon[dataset.content_type] = dataset

            # The cache is not just to save round-trips to the datastore, but to maximize reuse of decoded instances.
            self._cache[dataset] = {}

            # Make the dataset definitions easily accessible as attributes.
            self.__dict__[name] = dataset
        else:
            raise ValueError(f'Name collision: {name} is an existing attribute of Session')

    @property
    def client(self) -> pymongo.MongoClient:
        """The session's connection to the underlying datastore."""
        if self._client is None:
            self._client = pymongo.MongoClient(self._uri)
        return self._client

    def _cache_value(self, dataset: Dataset, doc) -> KbEntry:
        if doc['_id'] not in self._cache[dataset]:
            codec = dataset.codec or codecs.CODECS[dataset.content_type]
            self._cache[dataset][doc['_id']] = codec.decode(doc)
        return self._cache[dataset][doc['_id']]

    def get(self, dataset: Dataset, id) -> Optional[KbEntry]:
        """Retrieves the specified entry from the KB by ID, if it exists."""
        if id not in self._cache[dataset]:
            doc = self.client[dataset.db][dataset.collection].find_one(id)
            if doc:
                self._cache_value(dataset, doc)
        return self._cache[dataset].get(id)

    def put(self, dataset: Dataset, obj, bypass_cache=False):
        """Persists the object to the KB, in the given dataset."""
        if not bypass_cache:
            self._cache[dataset][obj.id] = obj
        else:
            # Even when bypassing the cache, make sure the cache itself is not now stale.
            self._cache[dataset].pop(obj.id, None)

        codec = dataset.codec or codecs.CODECS[dataset.content_type]
        doc = codec.encode(obj)
        self.client[dataset.db][dataset.collection].replace_one({'_id': obj.id}, doc, upsert=True)

    def find(self, dataset: Dataset, name: str, include_aka=True) -> List[KbEntry]:
        """Finds any number of KB entries matching the given name, optionally as an AKA."""
        found = set()
        docs = []
        for doc in self.client[dataset.db][dataset.collection].find(
                {'name': name}).collation({'locale': 'en', 'strength': 1}):
            if doc['_id'] not in found:
                docs.append(doc)
                found.add(doc['_id'])
        if include_aka:
            for doc in self.client[dataset.db][dataset.collection].find(
                    {'aka': name}).collation({'locale': 'en', 'strength': 1}):
                if doc['_id'] not in found:
                    docs.append(doc)
                    found.add(doc['_id'])
        return [self._cache_value(dataset, doc) for doc in docs]

    def xref(self, dataset: Dataset, q) -> List[KbEntry]:
        """Finds any number of entries in the dataset cross-referenced to the given query (DbXref or string)."""
        xref = self.as_xref(q)
        query = {'xrefs.id': xref.id}
        if xref.db:
            query['xrefs.db'] = xref.db

        results = []
        for doc in self.client[dataset.db][dataset.collection].find(query).collation({'locale': 'en', 'strength': 1}):
            results.append(self._cache_value(dataset, doc))
        return results

    def clear_cache(self, *datasets):
        """Clears cached objects for select datasets, or all datasets."""
        if not datasets:
            datasets = self.schema.values()
        for dataset in datasets:
            self._cache[dataset].clear()

    def as_xref(self, q) -> DbXref:
        """Attempts to coerce the query to a DbXref."""
        if isinstance(q, DbXref):
            return q
        elif isinstance(q, str):
            return DbXref.from_str(q)
        else:
            raise TypeError(f"{q} cannot be converted to DbXref.")

    def deref(self, q) -> Optional[KbEntry]:
        """Retrieves the object referred to by a DbXref or its string representation."""
        xref = self.as_xref(q)
        if xref.db in self.schema:
            return self.get(self.schema[xref.db], xref.id)
        else:
            return None

    def as_molecule(self, q) -> Optional[Molecule]:
        """Attempts to coerce the query to a Molecule."""
        if isinstance(q, Molecule):
            return q
        mol = self.deref(q)
        if not mol and Molecule in self.canon and isinstance(q, str):
            mol = self.get(self.canon[Molecule], q)
        return mol

    def as_reaction(self, q) -> Optional[Reaction]:
        """Attempts to coerce the query to a Reaction."""
        if isinstance(q, Reaction):
            return q
        rxn = self.deref(q)
        if not rxn and Reaction in self.canon and isinstance(q, str):
            rxn = self.get(self.canon[Reaction], q)
        return rxn

    def as_pathway(self, q) -> Optional[Pathway]:
        """Attempts to coerce the query to a Pathway."""
        if isinstance(q, Pathway):
            return q
        pw = self.deref(q)
        if not pw and Pathway in self.canon and isinstance(q, str):
            pw = self.get(self.canon[Pathway], q)
        return pw

    def __call__(self, q) -> Optional[KbEntry]:
        """Convenience interface to the KB.

        Attempts to return a unique object as intended by the caller. For unambiguous queries this works well, e.g.:
        - Fully specified DB:ID xref (as a DbXref or a string)
        - Unique ID of an object in a canonical dataset
        - ID of an object in any dataset, provided it is unique across all datasets

        This call will always return the first entry it finds, so if the conditions above do not hold, the result may
        not be as expected. CALLER BEWARE.

        This interface is provided to reduce friction in accessing KB entries interactively. Since multiple calls to the
        underlying DB may be needed to resolve any query, this is unlikely to be the most efficient way to access large
        numbers of objects. Consider using more specific methods as appropriate.

        Args:
            q: a KbEntry, DbXref, or string sufficient to identify a unique entry in the knowledge base.

        Returns:
            A single KbEntry identified by q, or None.
        """
        # Trivial exit
        if isinstance(q, KbEntry):
            return q

        # Fully specified xref
        entry = self.deref(q)
        if entry:
            return entry

        # ID in a canonical dataset
        for dataset in self.canon.values():
            entry = self.get(dataset, q)
            if entry:
                return entry

        # ID in a non-canonical dataset
        for dataset in self.schema.values():
            if self.canon.get(dataset.content_type) == dataset:
                continue  # already tried
            entry = self.get(dataset, q)
            if entry:
                return entry


class LookupCodec(codecs.Codec):
    """Session-aware Codec encoding a KbEntry by its ID, and decoding by looking it up in a given dataset."""
    def __init__(self, source, dataset):
        self._source = source
        self._dataset = dataset

    def encode(self, obj):
        return obj.id

    def decode(self, id):
        return self._source.get(self._dataset, id)


def configure_kb(uri=None):
    """Returns a Session object configured to access all reference and canonical KB datasets."""
    session = Session(uri=uri)

    # Reference datasets (local copies of external sources)
    session.define_dataset('EC', Dataset('ref', 'EC', KbEntry))
    session.define_dataset('GO', Dataset('ref', 'GO', KbEntry))
    session.define_dataset('CHEBI', Dataset('ref', 'CHEBI', Molecule))
    session.define_dataset('RHEA', Dataset('ref', 'RHEA', Reaction, codecs.ObjectCodec(Reaction, {
        'xrefs': codecs.ListCodec(item_codec=codecs.CODECS[DbXref], list_type=set),
        'stoichiometry': codecs.MappingCodec(key_codec=LookupCodec(session, session.CHEBI)),
        'catalyst': codecs.MOL_ID,
    })))

    # The KB proper - compiled, reconciled, integrated
    session.define_dataset('compounds', Dataset('kb', 'compounds', Molecule), canonical=True)
    session.define_dataset('reactions', Dataset('kb', 'reactions', Reaction, codecs.ObjectCodec(Reaction, {
        'xrefs': codecs.ListCodec(item_codec=codecs.CODECS[DbXref], list_type=set),
        'stoichiometry': codecs.MappingCodec(key_codec=LookupCodec(session, session.compounds)),
        'catalyst': codecs.MOL_ID,
    })), canonical=True)
    session.define_dataset('pathways', Dataset('kb', 'pathways', Pathway, codecs.ObjectCodec(Pathway, {
        'xrefs': codecs.ListCodec(item_codec=codecs.CODECS[DbXref], list_type=set),
        'metabolites': codecs.ListCodec(item_codec=LookupCodec(session, session.compounds)),
        'steps': codecs.ListCodec(item_codec=LookupCodec(session, session.reactions)),
        'enzymes': codecs.ListCodec(item_codec=codecs.MOL_ID),
    })), canonical=True)
    return session
