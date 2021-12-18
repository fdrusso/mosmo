"""Knowledge Base for Molecular Systems Modeling."""
from typing import Optional

import pymongo

from scheme import DbCrossRef, KbEntry, Molecule, Reaction, Specialization, Variation


class Connection:
    """Manages the connection to the MongoDB storage layer."""

    def __init__(self, uri: str = "mongodb://127.0.0.1:27017"):
        self._uri = uri
        self._client = None

    def connect(self, uri: Optional[str] = None):
        if uri is not None:
            self._uri = uri
        self._client = pymongo.MongoClient(self._uri)

    @property
    def client(self) -> pymongo.MongoClient:
        if self._client is None:
            self.connect()
        return self._client

    @property
    def compounds(self) -> pymongo.collection.Collection:
        return self.client.kb.compounds


KB = Connection()


# Defines the encoding/decoding schema for KB objects in the storage layer. PyMongo does have a mechanism to define
# custom types, but it is not flexible enough to do this cleanly in our case. The main cost is we need to define the
# schema relationships with explicit types, rather than having the system infer them. That is a reasonable constraint.

class AsIsCodec:
    """No-op codec passes everything through encode and decode as-is."""
    def encode(self, obj):
        return obj

    def decode(self, data):
        return data


class ObjectCodec:
    """Encodes/decodes a python instance to a json-compatible dict."""

    def __init__(self, clazz, codec_map=None, selective=False):
        self.clazz = clazz
        self.codec_map = codec_map or {}
        self.selective = selective

    def encode(self, obj):
        data = {}
        for k, v in obj.__dict__.items():
            if v is not None and (not self.selective or k in self.codec_map):
                data[k] = self.codec_map.get(k, AS_IS).encode(v)
        return data

    def decode(self, data):
        args = {}
        for k, v in data.items():
            if not self.selective or k in self.codec_map:
                args[k] = self.codec_map.get(k, AS_IS).decode(v)
        return self.clazz(**args)


class ListCodec:
    """Encodes/decodes a python iterable type to a json-compatible list."""

    def __init__(self, item_codec=None, list_type=list):
        self.list_type = list_type
        self.item_codec = item_codec or AS_IS

    def encode(self, items):
        return list(self.item_codec.encode(item) for item in items)

    def decode(self, data):
        return self.list_type(self.item_codec.decode(datum) for datum in data)


class MappingCodec:
    """Encodes/decodes a python mapping type to a json list of tuples."""

    def __init__(self, key_codec=None, value_codec=None, mapping_type=dict):
        self.mapping_type = mapping_type
        self.key_codec = key_codec or AS_IS
        self.value_code = value_codec or AS_IS

    def encode(self, mapping):
        return [[self.key_codec.encode(k), self.value_code.encode(v)] for k, v in mapping.items()]

    def decode(self, data):
        return self.mapping_type({self.key_codec.decode(k): self.value_code.decode(v) for k, v in data})


AS_IS = AsIsCodec()

CODECS = {
    DbCrossRef: ObjectCodec(DbCrossRef),
    Variation: ObjectCodec(Variation),
    Specialization: ObjectCodec(Specialization),
}

CODECS[KbEntry] = ObjectCodec(KbEntry, {
    "crossref": ListCodec(item_codec=CODECS[DbCrossRef])
})

CODECS[Molecule] = ObjectCodec(Molecule, {
    "crossref": ListCodec(item_codec=CODECS[DbCrossRef]),
    "variations": CODECS[Variation],
    "canonical_form": CODECS[Specialization],
    "default_form": CODECS[Specialization],
})

CODECS[Reaction] = ObjectCodec(Reaction, {
    "crossref": ListCodec(item_codec=CODECS[DbCrossRef]),
    "stoichiometry": MappingCodec(key_codec=ObjectCodec(Molecule, {"_id": AS_IS, "name": AS_IS}, selective=True)),
    "catalyst": ObjectCodec(Molecule, {"_id": AS_IS, "name": AS_IS}, selective=True),
})


def get_compound(compound_id) -> Optional[Molecule]:
    doc = KB.compounds.find_one(compound_id)
    if doc:
        return CODECS[Molecule].decode(doc)
    else:
        return None
