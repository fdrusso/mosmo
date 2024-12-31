"""Python (numpy) implementation of Schuster et al's algorithm to find elementary modes in reaction networks.

References:
- [Schuster _et al_ (2000)](https://www.nature.com/articles/nbt0300_326)
- [Schuster _et al_ (2002)](https://link.springer.com/article/10.1007/s002850200143)

The central data structure is a tableau, or composite matrix. The first num_mets (n) columns correspond to metabolites,
followed by num_rxns (m) columns corresponding to reactions. Each row represents one 'mode', where the right-hand m
columns specify a linear combination of reactions, and the left-hand n columns hold the resulting stoichiometry. The
initial tableau is simply the S matrix (transposed) coupled to an m by m identity matrix. The algorithm iteratively
updates this tableau, where iteration j ensures the first j metabolites are at steady state for all modes, and all modes
are 'elementary' with respect to those metabolites (see references for full definition).

An additional constraint is that any reaction may be designated irreversible, meaning it may not be used with a negative
coefficient in any mode.
"""
from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, List, Mapping, Optional, Set

import numpy as np

from mosmo.model import Molecule, Reaction, Pathway


@dataclass
class Mode:
    """Describes a reaction mode, i.e. a linear combination of reactions."""
    coefficients: Mapping[Reaction, int]
    reversible: bool

    @cached_property
    def description(self):
        parts = []
        for reaction, k in self.coefficients.items():
            if not k:
                continue

            if k < 0:
                parts.append('-')
            elif len(parts) > 0:
                parts.append('+')

            if abs(k) != 1:
                parts.append(str(abs(k)))

            parts.append(reaction.label)

        return ' '.join(parts)

    @cached_property
    def net_reaction(self):
        """A single Reaction describing the net effect of the linear combination represented by this mode."""
        return sum(rxn * k for rxn, k in self.coefficients.items())

    def __repr__(self):
        return self.description


def _sort_tableau(tableau, j):
    """Partitions the tableau into rows that are currently elementary for column j, and those that are not."""
    elementary = []
    pending = []
    for row in tableau:
        if row[0][j] == 0:
            elementary.append(row)
        else:
            pending.append(row)
    return elementary, pending


def _generate_candidates(pending, j):
    """Yields candidate pairs of rows that may be combined validly into a new row that eliminates metabolite j."""
    for i, (row_i, reversible_i, used_i) in enumerate(pending):
        for row_m, reversible_m, used_m in pending[i + 1:]:
            # Put a reversible mode second if possible, so we can always multiply the first by a positive number
            if reversible_m:
                yield row_i, row_m, reversible_i, used_i | used_m
            elif reversible_i:
                yield row_m, row_i, False, used_i | used_m

            # Otherwise we can still combine them, if they have opposite stoichiometry
            elif row_i[j] * row_m[j] < 0:
                yield row_i, row_m, False, used_i | used_m


def _merge_modes(row_i, row_m, reversible, j, num_rxns):
    """Performs the work of combining a pair of rows into a new row that eliminates metabolite j."""
    # All integer arithmetic.
    multiple = int(np.lcm(row_i[j], row_m[j]))
    # scale_i is always positive
    scale_i = int(multiple / abs(row_i[j]))
    # scale_m satisfies scale_i * mode_i[j] + scale_m * mode_m[j] = 0.
    scale_m = -int(scale_i * row_i[j] / row_m[j])

    # Combine rows, and reduce to simplest integers.
    row = scale_i * row_i + scale_m * row_m
    row = (row / np.gcd.reduce(row)).astype(int)

    # Determine the actual new used set for the merged row.
    used = set(np.nonzero(row[-num_rxns:])[0])

    # Mostly aesthetic, but prefer original reaction direction for reversible modes.
    if reversible:
        forward = np.nonzero(row[-num_rxns:] > 0)[0]
        if len(forward) * 2 < len(used):
            row = -row

    return row, reversible, used


def _process_candidates(candidates, elementary, j, num_rxns):
    """Decide which pairs of candidate rows to merge, and generate a new tableau."""
    # Every candidate must be compared against all current elementary modes, based on the non-subset zeros test.
    # Successful candidates extend the list of elementary modes that later candidates must be compared to.
    for row_i, row_m, reversible, used in candidates:
        passing = True
        for _, _, other_used in elementary:
            if used >= other_used:  # Superset or equal: the merged row would not be elementary.
                passing = False
                break

        # If the candidate survived, keep it. Other candidates must now compare against this new mode too.
        if passing:
            elementary.append(_merge_modes(row_i, row_m, reversible, j, num_rxns))

    return elementary


def elementary_modes(
        pathway: Pathway,
        intermediates: Set[Molecule],
        reversibility: Optional[Mapping[Reaction, bool]] = None,
        flipped: Optional[Iterable[Reaction]] = (),
) -> List[Mode]:
    """Main entry point for elementary mode algorithm.

    Args:
        pathway: The pathway of interest.
        intermediates: Molecules in `pathway` to be treated as intermediates, i.e. at steady state
            in any elementary mode of the pathway.
        reversibility: (optional) overrides intrinsic reversibility of any reaction.
        flipped: (optional) flips the direction of selected reactions, particularly if we need to treat them
            as irreversible in the opposite direction

    Returns:
        The list of elementary modes of the pathway, as defined in
        [Schuster _et al_ (2000)](https://www.nature.com/articles/nbt0300_326).
    """
    i_intermediates = [met in intermediates for met in pathway.molecules]
    s_matrix = pathway.s_matrix[i_intermediates].astype(int)
    num_mets, num_rxns = s_matrix.shape

    reversibility = reversibility or {}
    rxn_direction = np.ones(num_rxns, dtype=int)
    for rxn in flipped:
        rxn_direction[pathway.reactions.index_of(rxn)] = -1
    rows = np.concatenate([(s_matrix * rxn_direction).T, np.eye(num_rxns, dtype=int)], axis=1)

    # Internally we do not use an actual numpy matrix for the tableau, but rather a list of 1d rows with the same
    # structure. Each row has associated reversibility, plus the set of indices of all reactions included in that
    # mode. This is the complement of the set designated S(m_i) by Schuster et al.
    tableau = []
    for i, (rxn, row) in enumerate(zip(pathway.reactions, rows)):
        reversible = reversibility.get(rxn, rxn.reversible)
        tableau.append((row, reversible, {i}))

    # After iteration j, all columns up to j will be at steady state (i.e. 0), given the linear combination
    # of reactions represented in the right-hand num_rxns columns.
    for j in range(num_mets):
        elementary, pending = _sort_tableau(tableau, j)
        candidates = _generate_candidates(pending, j)
        tableau = _process_candidates(candidates, elementary, j, num_rxns)

    modes = []
    for row, reversible, _ in tableau:
        coefficients = row[-num_rxns:] * rxn_direction
        modes.append(
            Mode(
                coefficients={rxn: int(k) for rxn, k in zip(pathway.reactions, coefficients) if k != 0},
                reversible=reversible
            )
        )

    return modes
