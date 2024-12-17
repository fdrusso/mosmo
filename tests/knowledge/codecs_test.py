"""Tests for mosmo.knowledge.codecs."""
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from mosmo.knowledge import codecs


@dataclass
class _Base:
    _int: Optional[int] = None
    _float: Optional[float] = None
    _str: Optional[str] = None
    _volatile: Optional[str] = None  # Not encoded by any codec

    def __eq__(self, other):
        return (
                type(other) == type(self)
                and other._int == self._int
                and other._float == self._float
                and other._str == self._str
                and other._volatile == self._volatile
        )

    def __hash__(self):
        return hash((self._int, self._float, self._str, self._volatile))


@dataclass
class _Extended(_Base):
    _list: Optional[List] = None
    _set: Optional[Set] = None
    _dict: Optional[Dict] = None

    def __eq__(self, other):
        return (
                super().__eq__(other)
                and other._list == self._list
                and other._set == self._set
                and other._dict ==self._dict
        )

    def __hash__(self):
        return super().__hash__() + hash((self._list, self._set, self._dict))


BASE_CODEC = codecs.ObjectCodec(
    _Base,
    codec_map={
        '_int': codecs.AS_IS,
        '_float': codecs.AS_IS,
        '_str': codecs.AS_IS,
        # NOT _volatile
    }
)

EXTENDED_CODEC = codecs.ObjectCodec(
    _Extended,
    parent=BASE_CODEC,
    codec_map= {
        '_list': codecs.ListCodec(item_codec=BASE_CODEC),
        '_set': codecs.ListCodec(item_codec=BASE_CODEC, list_type=set),
        '_dict': codecs.MappingCodec(key_codec=codecs.AS_IS, value_codec=BASE_CODEC)
    }
)

class TestCodec:
    def test_ObjectCodec(self):
        # Test only the basic attributes first; defer list, set, and dict until we've tested the corresponding codecs.
        orig = _Base(
            _int=42,
            _float=3.14,
            _str='foobarbas',
        )
        codec = BASE_CODEC
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ObjectCodec_Rename(self):
        """Object members are renamed appropriately in encoded documents."""
        orig = _Base(
            _int=42,
            _float=3.14,
            _str='foobarbas',
        )
        codec = codecs.ObjectCodec(
            _Base,
            codec_map=BASE_CODEC.codec_map,
            rename={'_int': 'someint'})
        doc = codec.encode(orig)
        enc = json.dumps(doc)
        restored = codec.decode(json.loads(enc))
        assert doc['someint'] == orig._int
        assert restored == orig

    def test_ListCodec_Basic(self):
        orig = ['person', 'woman', 'man', 'camera', 'tv']
        codec = codecs.ListCodec()
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ListCodec_Deep(self):
        orig = [_Base(_int=117), _Base(_float=2.71828), _Base(_str='Hello World')]
        codec = codecs.ListCodec(item_codec=BASE_CODEC)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_SetCodec_Basic(self):
        orig = {'person', 'woman', 'man', 'camera', 'tv'}
        codec = codecs.ListCodec(list_type=set)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_SetCodec_Deep(self):
        orig = {_Base(_int=117), _Base(_float=2.71828), _Base(_str='Hello World')}
        codec = codecs.ListCodec(item_codec=BASE_CODEC, list_type=set)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ObjectCodec_Full(self):
        orig = _Extended(
            _int=42,
            _float=3.14,
            _str='foobarbas',
            _list=[_Base(_int=17), _Base(_float=2.71828), _Base(_str='Hello World')],
            _set={_Base(_str='foo'), _Base(_str='bar'), _Base(_int=-1)},
            _dict={'seventeen': _Base(_int=17), 'e': _Base(_float=2.71828), 'greeting': _Base(_str='Hello World')},
        )
        codec = EXTENDED_CODEC
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert orig == restored

    def test_ObjectCodec_Explicit_Only(self):
        orig = _Base(
            _str='here',
            _volatile='not'
        )
        codec = BASE_CODEC
        doc = codec.encode(orig)
        enc = json.dumps(doc)
        restored = codec.decode(json.loads(enc))
        assert '_volatile' not in doc
        assert orig != restored
