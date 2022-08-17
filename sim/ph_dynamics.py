"""Buffered pH Dynamics

With some inspiration from [Glaser _et al_ (2014)](https://doi.org/10.1021/ed400808c). The core approach
is to model the mass-action kinetics of proton dissociation/association in water, plus other sites as configured
with specified $pK_a$s.

All are modeled as _dissociation_ reactions, ${HA} \rightleftharpoons A^- + H^+$. So:
- $v_f = k_f [HA]$
- $v_b = k_b [H^+] [A^-]$
- $K_a = k_f / k_b$
- $pK_a = -log_{10}(K_a)$

where _f_ subscripts refer to the forward reaction (dissociation), and _b_ to back (association).

In general, $pK_a$ is measured, but $k_f$ and $k_b$ are not independently known. In the absence of other information
we assume a constant value for $k_b$, with the rationale that it is dominated by frequency of collision with $H^+$
(or $H_3O^+$).
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, optimize

# Molecular species represented by (name, charge)
Molecule = Tuple[str, int]
# All modeled events involve dissociation of a proton (H+).
PROTON = ('H', +1)


@dataclass
class AcidSet:
    """Represents a series of molecular species resulting from successive deprotonations off a single core.

    Attrs:
      species: The sequence of molecular species, each the result of deprotonation of the one before.
      p_kas: p_kas[i] is the pKa of the reaction where species[i] is the acid and species[i+1] is the base.
    """
    species: Sequence[Molecule]
    p_kas: Sequence[float]

    @staticmethod
    def for_acid(name: str, charge: int, p_kas: Iterable[float]):
        """Factory method for a sequence based on the same name, differentiated by charge."""
        species = [(f'{name}({charge:+d})', charge)]
        p_kas = list(p_kas)
        for i, p_ka in enumerate(p_kas):
            charge = charge - 1
            species.append((f'{name}({charge:+d})', charge))
        return AcidSet(species, p_kas)


# In an aqueous buffer, water itself serves as a proton donor, H2O <-> OH- + H+.
DISSOCIATION_OF_WATER = AcidSet([('H2O', 0), ('OH', -1)], [14.0])
# The second-order rate constant assumed for the reaction H+ + {base} -> {acid}. Units: per Molar per sec.
DEFAULT_KBACK = 1e9  # TODO: reference/justification for this value


class PhBuffer:
    """Models the dynamics of an aqueous buffer with multiple protonation sites."""
    def __init__(self, components: Iterable[AcidSet]):
        # Flattened list of all molecular species
        mols = [PROTON]
        # Flattened co=linear lists with the index of each acid and its pKa
        p_kas = []
        acids_idx = []

        for component in (DISSOCIATION_OF_WATER, *components):
            offset = len(mols)
            mols.extend(component.species)
            p_kas.extend(component.p_kas)
            acids_idx.extend(range(offset, offset + len(component.p_kas)))

        self.mols = mols
        kas = np.power(10, -np.array(p_kas))
        self.kf = kas * DEFAULT_KBACK
        self.kb = np.full_like(self.kf, DEFAULT_KBACK)

        # Reaction i is mols[acids_idx[i]] <-> mols[acids_idx[i] + 1] + PROTON
        self.acids_idx = np.array(acids_idx)
        self.bases_idx = self.acids_idx + 1

        # Use an S matrix just like any other reaction network.
        self.s_matrix = np.zeros((len(mols), len(kas)))
        cols = np.arange(self.s_matrix.shape[1])
        self.s_matrix[self.acids_idx, cols] = -1  # each reaction consumes the acid
        self.s_matrix[self.bases_idx, cols] = 1  # each reaction produces the base
        self.s_matrix[0, cols] = 1  # all reactions produce a proton

    def state_vector(self, concs: Mapping[Molecule, float], pH: float = 7.0) -> jnp.ndarray:
        values = [pow(10, -pH), 1, pow(10, pH - 14)]
        for mol in self.mols[3:]:
            values.append(concs.get(mol, 0))
        return jnp.array(values)

    def rates(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        vf = self.kf * y[self.acids_idx]
        vb = self.kb * y[0] * y[self.bases_idx]
        return vf, vb

    def s_dot_v(self, v: jnp.ndarray) -> jnp.ndarray:
        # An extensive attempt to optimize this based on the known structure actually ran ~50% slower.
        # Just trust jax to do this efficiently.
        return self.s_matrix @ v

    def dydt(self, y: jnp.ndarray) -> jnp.ndarray:
        vf, vb = self.rates(y)
        return self.s_dot_v(vf - vb)

    def equilibrium(self, concs: Mapping[Molecule, float], pH: float = 7.0, **kwargs) -> Mapping[Molecule, float]:
        """Find equilibrium from a given set of starting concentrations."""
        y0 = self.state_vector(concs, pH)

        def residual(x):
            """Deviation from steady state given a vector of net fluxes, x."""
            y = y0 + self.s_dot_v(x)
            dydt = self.dydt(y)
            return dydt

        soln = optimize.least_squares(
            fun=jax.jit(residual),
            jac=jax.jit(jax.jacfwd(residual)),
            x0=jnp.zeros_like(self.kf),
            **kwargs
        )
        y = y0 + self.s_dot_v(soln.x)
        return {m: v for m, v in zip(self.mols, np.asarray(y))}

    def titrate(self, concs: Mapping[Molecule, float], pH: float = 7.0, **kwargs) -> Mapping[Molecule, float]:
        """Find equilibrium from a given set of starting concentrations, holding pH constant."""
        y0 = self.state_vector(concs, pH)

        def residual(x):
            """Deviation from steady state given a vector of net fluxes, x."""
            # Hold pH constant by ignoring changes to [H+] itself.
            y = y0 + (self.s_dot_v(x)).at[0].set(0)
            dydt = self.dydt(y)
            return dydt

        soln = optimize.least_squares(
            fun=jax.jit(residual),
            jac=jax.jit(jax.jacfwd(residual)),
            x0=jnp.zeros_like(self.kf),
            **kwargs
        )
        y = y0 + (self.s_dot_v(jnp.asarray(soln.x))).at[0].set(0)
        return {m: v for m, v in zip(self.mols, np.asarray(y))}

    def simulate(self,
                 concs: Mapping[Molecule, float],
                 pH: float,
                 end: float,
                 step: float = 1e-7,
                 **kwargs):
        """Generate a timecourse of the dynamics of protonation/deprotonation from a given starting point."""
        fn = jax.jit(self.dydt)
        jac = jax.jit(jax.jacfwd(self.dydt))

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
