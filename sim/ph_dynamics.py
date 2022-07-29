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
we assume a constant value for $k_b$, with the rationale that it is dominated by frequency of collision with $H^+$.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy

# All modeled events involve dissociation of a proton (H+).
PROTON = ('H', +1)


@dataclass
class AcidBaseSequence:
    """Represents a sequence of successive deprotonations of a single core molecule.

    Attrs:
      species: The sequence of molecular species, each the result of deprotonation of the one before.
      p_kas: p_kas[i] is the pKa of the reaction where species[i] is the acid and species[i+1] is the base.
    """
    species: Sequence[Tuple[str, int]]
    p_kas: Sequence[float]

    @staticmethod
    def from_acid(name: str, charge: int, p_kas: Iterable[float]):
        """Factory method for a sequence based on the same name, differentiated by charge."""
        species = [(f'{name}({charge:+d})', charge)]
        p_kas = list(p_kas)
        for i, p_ka in enumerate(p_kas):
            charge = charge - 1
            species.append((f'{name}({charge:+d})', charge))
        return AcidBaseSequence(species, p_kas)


# In an aqueous buffer, water itself serves as a proton donor, H2O <-> OH- + H+.
DISSOCIATION_OF_WATER = AcidBaseSequence([('H2O', 0), ('OH', -1)], [14.0])
# The second-order rate constant assumed for the reaction H+ + {base} -> {acid}. Units: per Molar per sec.
DEFAULT_KBACK = 1e9  # TODO: reference/justification for this value


class PhBuffer:
    """Models the dynamics of an aqueous buffer with multiple protonation sites."""
    def __init__(self, components: Iterable[AcidBaseSequence]):
        mols = [PROTON]
        acids_idx = []
        bases_idx = []
        kas = []

        for component in (DISSOCIATION_OF_WATER, *components):
            start = len(mols)
            for i, (mol, p_ka) in enumerate(zip(component.species, component.p_kas)):
                mols.append(mol)
                acids_idx.append(start + i)
                bases_idx.append(start + i + 1)
                kas.append(pow(10, -p_ka))
            mols.append(component.species[-1])

        self.mols = mols
        self.acids_idx = np.array(acids_idx)
        self.bases_idx = np.array(bases_idx)

        self.kf = np.array(kas) * DEFAULT_KBACK
        self.kb = np.full_like(self.kf, DEFAULT_KBACK)

        # Use an S matrix just like any other reaction network. TODO: slicker ops eploiting known sparsity?
        self.s_matrix = np.zeros((len(mols), len(kas)))
        cols = np.arange(self.s_matrix.shape[1])
        self.s_matrix[self.acids_idx, cols] = -1  # all dissociations consume the acid
        self.s_matrix[self.bases_idx, cols] = 1  # all dissociations produce the base
        self.s_matrix[0, cols] = 1  # all dissociations produce a proton

    def state_vector(self, pH, concs):
        values = [pow(10, -pH), 1, pow(10, pH - 14)]
        for mol in self.mols[3:]:
            values.append(concs.get(mol, 0))
        return jnp.array(values)

    def rates(self, y):
        vf = self.kf * y[self.acids_idx]
        vb = self.kb * y[0] * y[self.bases_idx]
        return vf, vb

    def dydt(self, y):
        vf, vb = self.rates(y)
        vnet = vf - vb
        return self.s_matrix @ vnet

    def titrate(self, pH, concs, **kwargs):
        y0 = self.state_vector(pH, concs)

        def project_state(x):
            dissociations = self.s_matrix @ x
            y = y0 + dissociations
            return y.at[0].set(y0[0])

        def residual(x):
            y = project_state(x)
            dydt = self.dydt(y)
            return dydt

        soln = scipy.optimize.least_squares(
            fun=jax.jit(residual),
            jac=jax.jit(jax.jacfwd(residual)),
            x0=jnp.zeros(self.s_matrix.shape[1]),
            **kwargs
        )

        result = {m: v for m, v in zip(self.mols, np.asarray(project_state(soln.x)))}
        return result, soln

    def simulate(self, pH, concs, end, step=1e-7):
        fn = jax.jit(self.dydt)
        jac = jax.jit(jax.jacfwd(self.dydt))

        return scipy.integrate.solve_ivp(
            fun=lambda _, y: fn(y),
            jac=lambda _, y: jac(y),
            y0=self.state_vector(pH, concs),
            t_span=(0, end),
            t_eval=np.linspace(0, end, int(end / step) + 1),
            method='BDF',
            first_step=1e-9,  # pH is fast
        )
