"""Knowledge Base for Molecular Systems Modeling."""
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import pymongo

import knowledge.codecs as codecs
from model.core import DbXref, KbEntry, Molecule, Reaction, Pathway


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
        self._cache: Dict[Dataset, Dict[Any, KbEntry]] = {}

        if schema:
            for name, dataset in schema.items():
                self.define_dataset(name, dataset)

    def define_dataset(self, name: str, dataset: Dataset):
        """Adds a dataset to the schema of this session."""
        if name not in self.schema and name not in self.__dict__:
            self.schema[name] = dataset

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

    def _cache_value(self, dataset: Dataset, doc):
        if doc['_id'] not in self._cache[dataset]:
            codec = dataset.codec or codecs.CODECS[dataset.content_type]
            self._cache[dataset][doc['_id']] = codec.decode(doc)
        return self._cache[dataset][doc['_id']]

    def get(self, dataset, id):
        """Retrieves the specified entry from the KB by ID, if it exists."""
        if id not in self._cache[dataset]:
            doc = self.client[dataset.db][dataset.collection].find_one(id)
            if doc:
                self._cache_value(dataset, doc)
        return self._cache[dataset].get(id)

    def put(self, dataset, obj, bypass_cache=False):
        """Persists the object to the KB, in the given dataset."""
        if not bypass_cache:
            self._cache[dataset][obj.id] = obj
        else:
            # Even when bypassing the cache, make sure the cache itself is not now stale.
            self._cache[dataset].pop(obj.id, None)

        codec = dataset.codec or codecs.CODECS[dataset.content_type]
        doc = codec.encode(obj)
        self.client[dataset.db][dataset.collection].replace_one({'_id': obj.id}, doc, upsert=True)

    def find(self, dataset, name, include_aka=True):
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

    def xref(self, dataset, xref_id, xref_db=None):
        """Finds any number of KB entries cross-referenced to the given ID."""
        query = {'xrefs.id': xref_id}
        if xref_db:
            query['xrefs.db'] = xref_db

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
    """Returns a Session object configured to access all reference and KB datasets."""
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
    session.define_dataset('compounds', Dataset('kb', 'compounds', Molecule))
    session.define_dataset('reactions', Dataset('kb', 'reactions', Reaction, codecs.ObjectCodec(Reaction, {
        'xrefs': codecs.ListCodec(item_codec=codecs.CODECS[DbXref], list_type=set),
        'stoichiometry': codecs.MappingCodec(key_codec=LookupCodec(session, session.compounds)),
        'catalyst': codecs.MOL_ID,
    })))
    session.define_dataset('pathways', Dataset('kb', 'pathways', Pathway, codecs.ObjectCodec(Pathway, {
        'xrefs': codecs.ListCodec(item_codec=codecs.CODECS[DbXref], list_type=set),
        'metabolites': codecs.ListCodec(item_codec=LookupCodec(session, session.compounds)),
        'steps': codecs.ListCodec(item_codec=LookupCodec(session, session.reactions)),
        'enzymes': codecs.ListCodec(item_codec=codecs.MOL_ID),
    })))
    return session
