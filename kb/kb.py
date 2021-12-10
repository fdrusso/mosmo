"""Knowledge Base for Molecular Systems Modeling."""
from typing import Optional

import pymongo

from scheme import DbCrossRef, Molecule


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


class ObjectCodec:
    """Encodes/decodes a python instance to a json-compatible dict."""

    def __init__(self, clazz, codec_map=None):
        self.clazz = clazz
        self.codec_map = codec_map or {}

    def encode(self, obj):
        data = {}
        for k, v in obj.__dict__.items():
            if v is not None:
                if k in self.codec_map:
                    data[k] = self.codec_map[k].encode(v)
                else:
                    data[k] = v
        return data

    def decode(self, data):
        args = {}
        for k, v in data.items():
            if k in self.codec_map:
                args[k] = self.codec_map[k].decode(v)
            else:
                args[k] = v
        return self.clazz(**args)


class ListCodec:
    """Encodes/decodes a python iterable type to a json-compatible list."""

    def __init__(self, list_type=list, item_codec=None):
        self.list_type = list_type
        self.item_codec = item_codec

    def encode(self, items):
        if self.item_codec:
            return list(self.item_codec.encode(item) for item in items)
        else:
            return list(items)

    def decode(self, data):
        if self.item_codec:
            return self.list_type(self.item_codec.decode(datum) for datum in data)
        else:
            return self.list_type(data)


CODECS = {
    Molecule: ObjectCodec(Molecule, {"aka": ListCodec(), "crossref": ListCodec(list, ObjectCodec(DbCrossRef))}),
}


def get_compound(compound_id) -> Optional[Molecule]:
    doc = KB.compounds.find_one(compound_id)
    if doc:
        return CODECS[Molecule].decode(doc)
    else:
        return None
