"""Tests for mosmo.knowledge.codecs."""
from dataclasses import dataclass
import json
from typing import Dict, List, Optional, Set

from mosmo.knowledge import codecs


@dataclass
class _Obj:
    _int: Optional[int] = None
    _float: Optional[float] = None
    _str: Optional[str] = None
    _list: Optional[List] = None
    _set: Optional[Set] = None
    _dict: Optional[Dict] = None

    def __eq__(self, other):
        def opt_equal(a, b):
            return (a is None and b is None) or a == b

        return (
                type(other) == type(self)
                and opt_equal(other._int, self._int)
                and opt_equal(other._float, self._float)
                and opt_equal(other._str, self._str)
                and opt_equal(other._list, self._list)
                and opt_equal(other._set, self._set)
                and opt_equal(other._dict, self._dict)
        )

    def __hash__(self):
        return hash((self._int, self._float, self._str, self._list, self._set, self._dict))


class TestCodec:
    def test_ObjectCodec(self):
        # Test only the basic attributes first; defer list, set, and dict until we've tested the corresponding codecs.
        orig = _Obj(
            _int=42,
            _float=3.14,
            _str="foobarbas",
        )
        codec = codecs.ObjectCodec(_Obj)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ListCodec_Basic(self):
        orig = ["person", "woman", "man", "camera", "tv"]
        codec = codecs.ListCodec()
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ListCodec_Deep(self):
        orig = [_Obj(_int=117), _Obj(_float=2.71828), _Obj(_str="Hello World")]
        codec = codecs.ListCodec(codecs.ObjectCodec(_Obj))
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_SetCodec_Basic(self):
        orig = {"person", "woman", "man", "camera", "tv"}
        codec = codecs.ListCodec(list_type=set)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_SetCodec_Deep(self):
        orig = {_Obj(_int=117), _Obj(_float=2.71828), _Obj(_str="Hello World")}
        codec = codecs.ListCodec(codecs.ObjectCodec(_Obj), list_type=set)
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert restored == orig

    def test_ObjectCodec_Full(self):
        orig = _Obj(
            _int=42,
            _float=3.14,
            _str="foobarbas",
            _list=[_Obj(_int=117), _Obj(_float=2.71828), _Obj(_str="Hello World")],
            _set={"person", "woman", "man", "camera", "tv"},
            _dict={_Obj(_int=117): 17, _Obj(_float=2.71828): 255, _Obj(_str="Hello World"): 0},
        )
        codec = codecs.ObjectCodec(
            clazz=_Obj,
            codec_map={
                "_list": codecs.ListCodec(codecs.ObjectCodec(_Obj)),
                "_set": codecs.ListCodec(list_type=set),
                "_dict": codecs.MappingCodec(key_codec=codecs.ObjectCodec(_Obj), value_codec=codecs.AS_IS)
            }
        )
        enc = json.dumps(codec.encode(orig))
        restored = codec.decode(json.loads(enc))
        assert orig == restored
