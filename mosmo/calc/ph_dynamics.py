"""Buffered pH Dynamics

With some inspiration from [Glaser _et al_ (2014)](https://doi.org/10.1021/ed400808c). The core approach
is to model the mass-action kinetics of proton dissociation/association in water, plus any number of protonation sites
with specified $pK_a$s.

All are modeled as _dissociation_ reactions, ${HA} \rightleftharpoons A^- + H^+$. So:
- $v_f = k_f [HA]$
- $v_b = k_b [H^+] [A^-]$
- $K_a = k_f / k_b$
- $pK_a = -log_{10}(K_a)$

where _f_ subscripts refer to the forward reaction (dissociation), and _b_ to back (association).

In general, $pK_a$ is measured, but $k_f$ and $k_b$ are not independently known. In the absence of other information
we assume a constant value for $k_b$, with the rationale that it is dominated by the frequency of collision with $H^+$
(or $H_3O^+$).
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, optimize

from mosmo.model import Molecule

# Built-in definitions for key components, avoids dependence on any specific KB sources. If desired, these can safely
# be replaced with application-specific definitions before using PhBuffer.
PROTON = Molecule(id='PROTON', name='proton', shorthand='H+', charge=+1)
HYDROXYL = Molecule(id='HYDROXYL', name='hydroxyl group', shorthand='OH-', charge=-1)
WATER = Molecule(id='WATER', name='water', shorthand='H2O', charge=0)

# The second-order rate constant assumed for all reactions H+ + {base} -> {acid}. Units: per Molar per sec.
DEFAULT_KBACK = 1e9  # TODO: reference/justification for this value


@dataclass
class ProtonationSequence:
    """Represents a series of molecular species resulting from sequential protonation of a single core.

    Attrs:
      species: The sequence of molecular species, from most basic to most acidic.
      p_kas: p_kas[i] is the pKa of the reaction where species[i] is the base and species[i+1] is the acid.
    """
    species: Sequence[Molecule]
    p_kas: Sequence[float]


class PhBuffer:
    """Models the dynamics of an aqueous buffer with components holding one or more protonation sites.

    Conceptually this is a type of reaction network, with a strictly defined structure where each 'reaction' is the
    reversible dissociation of a proton at a single site. Conventionally each site is described by a single constant,
    pKa, the simplest interpretation of which is the pH where the site is half-protonated. Internally, this class goes
    one step further to describe each reaction using forward (dissociation) and reverse (association) kinetic rate
    constants.

    Since this models aqueous buffers specifically, the dissociation of water is always part of the system.

    Attrs:
        species: all distinct species in the system, in various protonation states. The order is maintained for state
            and dynamics vectors, and any operations involving the molecular constituents of the buffer system. The
            first three species are hardcoded to be protons, hydroxyl ions, and water.
        acids: the index in `species` of the acid (protonated) form for each protonation site.
        bases: the index in `species` of the base (deprotonated) form for each protonation site.
        kf: forward rate constant, for dissociation (first-order) of the proton at each protonation site.
        kb: back rate constant, for association (second-order) of the proton at each protonation site.
    """

    def __init__(self, components: Iterable[ProtonationSequence]):
        species = [PROTON, HYDROXYL, WATER]
        bases = [1]  # i.e. base=HYDROXYL, acid=WATER
        p_kas = [14.]
        offset = 3

        for component in components:
            species.extend(component.species)
            p_kas.extend(component.p_kas)
            bases.extend(range(offset, offset + len(component.p_kas)))
            offset = offset + len(component.species)

        self.species = species
        self.bases = np.array(bases)
        self.acids = self.bases + 1

        kas = np.power(10, -np.array(p_kas))
        self.kf = kas * DEFAULT_KBACK
        self.kb = np.full_like(self.kf, DEFAULT_KBACK)

        # Use an S matrix just like any other reaction network. Note an extensive attempt to skip the S matrix and
        # optimize calculations based on the known structure actually ran ~50% slower. Stick with the matrix math.
        self.s_matrix = np.zeros((len(species), len(kas)))
        cols = np.arange(self.s_matrix.shape[1])
        self.s_matrix[self.acids, cols] = -1  # each dissociation consumes the acid
        self.s_matrix[self.bases, cols] = 1  # each dissociation produces the base
        self.s_matrix[0, cols] = 1  # each dissociation produces a proton
        self.s_matrix[2, cols] = 0  # nothing affects the constant activity of the solvent

    def state_vector(self, concs: Mapping[Molecule, float], pH: float) -> jnp.ndarray:
        """Packs a lookup of molecule concentrations into an array, for efficient calculations.

        Args:
            concs: Concentrations of molecules to be represented on the state vector. Any molecules not included are
                assumed to have a concentration of 0
            pH: the current pH of the system. Determines concentrations of the first two species, H+ and OH-.

        Returns:
            An array collinear with self.species
        """
        values = [concs.get(species, 0) for species in self.species]
        # H+ and OH- are determined by pH. Water (solvent) has constant activity set to 1.
        values[:3] = [pow(10, -pH), pow(10, pH - 14), 1]
        return jnp.array(values)

    def rates(self, state: jnp.ndarray) -> jnp.ndarray:
        """Instantaneous rates of all dissociation reactions, given a state vector."""
        h_conc = state[0]
        vf = self.kf * state[self.acids]
        vb = self.kb * h_conc * state[self.bases]
        return vf - vb

    def dstate_dt(self, rates: jnp.ndarray) -> jnp.ndarray:
        """Calculates the net effect of a vector of reaction rates on each species in the system."""
        dstate_dt = self.s_matrix @ rates
        return dstate_dt

    def equilibrium(self, concs: Mapping[Molecule, float], pH: float = 7.0, **kwargs) -> Mapping[Molecule, float]:
        """Find equilibrium from a given set of starting concentrations."""
        state0 = self.state_vector(concs, pH)

        def residual(x):
            # x is a vector of dissociations at each site; i.e. convert x[i] molecules of acids[i] into x[i]
            # molecules of bases[i] plus x[i] protons.
            state = state0 + self.dstate_dt(x)
            # At steady state, all dstate_dt values are zero
            return self.dstate_dt(self.rates(state))

        soln = optimize.least_squares(
            fun=jax.jit(residual),
            jac=jax.jit(jax.jacfwd(residual)),
            x0=jnp.zeros_like(self.kf),
            **kwargs
        )
        state = state0 + self.dstate_dt(soln.x)
        return dict(zip(self.species, np.asarray(state)))

    def titrate(self, concs: Mapping[Molecule, float], pH: float, **kwargs) -> Mapping[Molecule, float]:
        """Find equilibrium from a given set of starting concentrations, holding pH constant."""
        state0 = self.state_vector(concs, pH)

        def residual(x):
            # x is a vector of dissociations at each site, but holding protons constant. So convert x[i] molecules of
            # acids[i] into x[i] molecules of bases[i] but ignore changes to [H+] itself.
            state = state0 + self.dstate_dt(x).at[0].set(0)
            # At steady state, all dstate_dt values are zero
            return self.dstate_dt(self.rates(state))

        soln = optimize.least_squares(
            fun=jax.jit(residual),
            jac=jax.jit(jax.jacfwd(residual)),
            x0=jnp.zeros_like(self.kf),
            **kwargs
        )
        state = state0 + self.dstate_dt(jnp.asarray(soln.x)).at[0].set(0)
        return dict(zip(self.species, np.asarray(state)))

    def simulate(self,
                 concs: Mapping[Molecule, float],
                 pH: float,
                 end: float,
                 step: float = 1e-7,
                 **kwargs):
        """Generate a timecourse of the dynamics of protonation/deprotonation from a given starting point."""

        def dynamics(state):
            return self.dstate_dt(self.rates(state))

        fn = jax.jit(dynamics)
        jac = jax.jit(jax.jacfwd(dynamics))

        return integrate.solve_ivp(
            fun=lambda _, y: fn(y),
            jac=lambda _, y: jac(y),
            y0=self.state_vector(concs, pH),
            t_span=(0, end),
            t_eval=np.linspace(0, end, int(end / step) + 1),
            method='BDF',
            first_step=1e-9,  # pH is fast
            **kwargs
        )
