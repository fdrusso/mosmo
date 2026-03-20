"""Core classes defining objects and concepts used to construct models of molecular systems."""
import collections
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple, Union

from .base import DbXref, KbEntry, Registry


class VariableFeature:
    """Describe a point of variation on a molecule, as a choice among variant forms.

    Features and variants extend the is_a relationship used in many ontologies, by adding addressability. We don't just
    declare that <child> is_a <parent>, but that <child> is _the_ [foo, bar] form of the parent. As a practical example
    we may have an entry in our KB for glucose. We also know that glucose has D and L stereoisomers, and that because
    of ring-chain tautomerism, a given molecule may be in the open-chain, α, or β configurations. So, glucose is the
    general parent, and β-D-glucose is the [D, β] form of glucose.
    """

    class Variant:
        """Encapsulates a single variant of a VariableFeature."""

        def __init__(self, feature: "VariableFeature", label: str):
            self.feature = feature
            self.label = label

        def __repr__(self):
            return f"{self.feature.label}.{self.label}"

    def __init__(self, label, variants: Iterable[Union[str, Tuple[str, str]]]):
        self.label = label
        self._variants = {}
        for arg in variants:
            if isinstance(arg, tuple):
                vlabel, vattr = arg
            else:
                vlabel = arg
                vattr = arg.upper()

            variant = VariableFeature.Variant(self, vlabel)
            self._variants[vlabel] = variant
            setattr(self, vattr, variant)

    def get(self, vlabel: str):
        """Looks up a variant by its label."""
        return self._variants.get(vlabel)

    def __iter__(self):
        return iter(self._variants.values())

    def __repr__(self):
        return f"<{self.__class__.__name__}>{self.label}"

FEATURE_TYPES = Registry(VariableFeature, 'label')


@dataclass
class Molecule(KbEntry):
    """A molecule or molecule-like entity that may participate in a molecular system."""
    formula: Optional[str] = None
    """Chemical formula of this molecule."""

    mass: Optional[float] = None
    """Mass of one molecule, in daltons (or of a mole, in grams)."""

    charge: Optional[int] = None
    """Electric charge of the molecule."""

    structure: Optional[str] = None
    """SMILES string describing the structure, if available."""

    canonical_form: Optional[str] = None
    """The canonical parent of this molecule."""

    form_name: Optional[Tuple] = None
    """This molecule's relationship to its canonical parent, as a series of feature variants."""

    features: Optional[Mapping[str, VariableFeature]] = None
    """Defines the ways in which molecules of this type may vary."""

    child_forms: Optional[Mapping[Tuple, str]] = None
    """Keeps track of defined child forms, each identified by a tuple of feature variants."""

    default_form: Optional[str] = None
    """For a general (canonical) molecule, denotes a more specific assumed form under physiological conditions.

    As a specific example, we most often refer simply to ATP. But ATP technically has multiple protonation
    states, with slightly different mass and different charge. For simplicity we continue to refer simply 
    to ATP, but define that its default form is ATP [4-].
    """

    def _data_items(self):
        def form_name(form_tuple):
            return f"({', '.join(variant.label for variant in form_tuple or [])})"

        items = dict(super()._data_items())
        items.update({
            'formula': self.formula,
            'mass': self.mass,
            'charge': self.charge,
            'structure': self.structure,
        })

        if self.canonical_form:
            items['form_info'] = f'{form_name(self.form_name)} form of {self.canonical_form}'
        if self.features:
            items['features'] = ', '.join(f'{label}[{feature.label}]' for label, feature in self.features.items())
        if self.child_forms:
            child_info = []
            for form_tuple, child in self.child_forms.items():
                child_info.append(f'{form_name(form_tuple)}: {child}')
            items['child_forms'] = child_info
        if self.default_form:
            items['default_form'] = str(self.default_form)
        return items

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.name or ''}"


@dataclass
class Reaction(KbEntry):
    """A process transforming one set of molecules into another set of molecules in defined proportions."""
    stoichiometry: Mapping[Molecule, float] = None
    """The molecules transformed by this reaction. Substrates have negative stoichiometry, products positive."""

    catalyst: Optional[Molecule] = None
    """A single molecule (though possibly a complex) catalyzing this reaction. Neither consumed nor produced."""

    reversible: bool = True
    """Whether or not this reaction should be treated as reversible"""

    @property
    def equation(self):
        """Human-readable compact summary of the reaction."""

        def molecule_term(molecule: Molecule, count: float) -> str:
            if count == 1:
                return molecule.label
            else:
                return f'{count} {molecule.label}'

        lhs = [molecule_term(molecule, -count) for molecule, count in self.stoichiometry.items() if count < 0]
        rhs = [molecule_term(molecule, count) for molecule, count in self.stoichiometry.items() if count > 0]
        arrow = ' <=> ' if self.reversible else ' => '

        return ' + '.join(lhs) + arrow + ' + '.join(rhs)

    def _data_items(self):
        return super()._data_items() | {
            'equation': self.equation,
            'reversible': self.reversible,
            'catalyst': self.catalyst,
        }

    def __eq__(self, other):
        return self.same_as(other)

    def __hash__(self):
        return hash((type(self), self.id))

    def __repr__(self):
        return f"[{self.id}] {self.equation}"

    def __add__(self, other):
        """Combines this reaction with another."""
        # Trick to support sum(): Adding any kind of 0 is supported
        if not other:
            return self
        if not isinstance(other, Reaction):
            raise ValueError(f"Reaction cannot be combined with type [{type(other)}]")

        stoichiometry = collections.Counter()
        stoichiometry.update(self.stoichiometry)
        stoichiometry.update(other.stoichiometry)
        return Reaction(
            id=self.id + "+" + other.id,
            db=None,
            stoichiometry={molecule: count for molecule, count in stoichiometry.items() if count != 0},
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (other * -1)

    def __mul__(self, other):
        """Multiplies the effect of this reaction proportionally across all reactants."""
        if not isinstance(other, (int, float)):
            raise ValueError(f"Reaction cannot be multiplied by type [{type(other)}]")

        return Reaction(
            id=str(other) + "*" + self.id,
            db=None,
            stoichiometry={molecule: other * count for molecule, count in self.stoichiometry.items()},
        )

    __rmul__ = __mul__
