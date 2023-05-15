"""Tests for mosmo.model.reaction_network."""
from dataclasses import dataclass

import numpy as np

from mosmo.model.base import KbEntry
from mosmo.model.core import Molecule, Reaction
from mosmo.model.reaction_network import Index, ReactionNetwork


@dataclass
class _Tidbit(KbEntry):
    """KbEntry subclass used for testing."""

    def __eq__(self, other):
        return type(self) == type(other) and self.id == other.id

    def __hash__(self):
        return hash((type(self), self.id))


T1 = _Tidbit("t1", description="Your mother was a hamster")
T2 = _Tidbit("t2", description="Your father smelt of elderberries")
T3 = _Tidbit("t3", description="We already have one")


class TestIndex:
    def test_ListLike(self):
        """Tests the list-like behavior of Index."""
        tidbits: Index[_Tidbit] = Index([T1, T2])
        assert len(tidbits) == 2
        assert T1 in tidbits
        for i, tidbit in enumerate(tidbits):
            assert tidbit is tidbits[i]
        assert tidbits.index_of(T2) == 1

    def test_SetLike(self):
        """Tests the set-like behavior of Index."""
        tidbits: Index[_Tidbit] = Index()
        assert len(tidbits) == 0

        tidbits.add(T1)
        assert len(tidbits) == 1
        assert T1 in tidbits
        assert T2 not in tidbits

        tidbits.add(T2)
        tidbits.add(T1)  # No-op by set semantics
        assert len(tidbits) == 2
        assert tidbits[-1] is T2

        tidbits.update([T2, T3])
        assert len(tidbits) == 3
        assert tidbits.index_of(T3) == 2

    def test_Pack(self):
        tidbits: Index[_Tidbit] = Index([T1, T2, T3])
        v = tidbits.pack({T2: 3}, default=-1)
        assert np.all(v == np.array([-1, 3, -1]))

    def test_Unpack(self):
        tidbits: Index[_Tidbit] = Index([T2, T3, T1])  # Changed order
        data = tidbits.unpack(np.array([3.14159, 2.71828, 1.618]))
        assert data[T1] == 1.618
        assert data[T2] == 3.14159
        assert data[T3] == 2.71828


ABCD = Reaction("abcd", stoichiometry={Molecule("a"): -1, Molecule("b"): -2, Molecule("c"): 2, Molecule("d"): 1})
BDE = Reaction("bde", stoichiometry={Molecule("b"): -1, Molecule("d"): -1, Molecule("e"): 2})


class TestReactionNetwork:
    def test_Shape(self):
        """A network has the expected shape."""
        network = ReactionNetwork([ABCD, BDE])
        unique_reactants = set(m for r in [ABCD, BDE] for m in r.stoichiometry)
        assert network.shape == (len(unique_reactants), 2)

    def test_SMatrix(self):
        """The s_matrix matches the stoichiometry of the input reactions."""
        network = ReactionNetwork([ABCD, BDE])
        for i, m in enumerate(network.reactants):
            for j, r in enumerate(network.reactions):
                assert network.s_matrix[i, j] == r.stoichiometry.get(m, 0)
