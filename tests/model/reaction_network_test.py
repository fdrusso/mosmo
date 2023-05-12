"""Tests for mosmo.model.reaction_network."""
from dataclasses import dataclass

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


class TestIndex:
    def test_ListLike(self):
        """Tests the list-like behavior of Index."""
        x = _Tidbit("x", description="Your mother was a hamster")
        y = _Tidbit("y", description="Your father smelt of elderberries")
        tidbits = Index[_Tidbit]([x, y])
        assert len(tidbits) == 2
        assert x in tidbits
        for i, tidbit in enumerate(tidbits):
            assert tidbit is tidbits[i]
        assert tidbits.index_of(y) == 1

    def test_SetLike(self):
        """Tests the set-like behavior of Index."""
        x = _Tidbit("x", description="Your mother was a hamster")
        y = _Tidbit("y", description="Your father smelt of elderberries")
        z = _Tidbit("z", description="We already have one")

        tidbits = Index[_Tidbit]()
        assert len(tidbits) == 0

        tidbits.add(x)
        assert len(tidbits) == 1
        assert x in tidbits
        assert y not in tidbits

        tidbits.add(y)
        tidbits.add(x)  # No-op by set semantics
        assert len(tidbits) == 2
        assert tidbits[-1] is y

        tidbits.update([y, z])
        assert len(tidbits) == 3
        assert tidbits.index_of(z) == 2


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
