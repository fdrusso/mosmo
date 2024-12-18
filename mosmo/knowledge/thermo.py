"""Thermodynamics via equilibrator_api."""
from typing import List, Mapping, Optional

import equilibrator_api
import numpy as np

from mosmo.model import DS, Molecule, Reaction


class Thermodynamics:
    """Adapter class to use equilibrator_api to find thermodynamic properties of Molecules and Reactions."""
    def __init__(self, p_h=7.3, p_mg=1.5, ionic_strength=0.25, temperature=298.15):
        self.cc = equilibrator_api.ComponentContribution()
        self.cc.p_h = equilibrator_api.Q_(p_h)
        self.cc.p_mg = equilibrator_api.Q_(p_mg)
        self.cc.ionic_strength = equilibrator_api.Q_(f"{ionic_strength}M")
        self.cc.temperature = equilibrator_api.Q_(f"{temperature}K")

        self._cc_compounds = {}
        self._cc_reactions = {}
        self._dg_f = {}

    def cc_compound(self, molecule: Molecule):
        if molecule not in self._cc_compounds:
            # Sources supported by eQuilibrator, as documented at
            # https://equilibrator.readthedocs.io/en/latest/tutorial.html#creating-a-compound-object
            xrefs = {xref.db: xref for xref in molecule.xrefs or []}
            found = None
            for ds, prefix in [
                (DS.KEGG, "kegg:"),
                (DS.METACYC, "metacyc.compound:"),
                (DS.METANETX, "metanetx.chemical:"),

                # eQ appears to get confused by CHEBI. Stick to the general forms above.
                # (DS.CHEBI, "chebi:CHEBI:"),
                # Issue with BIGG ids - eQ does not recognize the _c or _e suffix
                # (DS.BIGG, "bigg.metabolite:"),
            ]:
                if ds in xrefs:
                    found = self.cc.get_compound(prefix + xrefs[ds].id)
                    if found:
                        break
            self._cc_compounds[molecule] = found  # Do not retry on future calls, even if unmatched.
        return self._cc_compounds[molecule]

    def cc_reaction(self, reaction: Reaction) -> Optional[equilibrator_api.Reaction]:
        if reaction not in self._cc_reactions:
            stoich = {}
            for molecule, count in reaction.stoichiometry.items():
                cc_compound = self.cc_compound(molecule)
                if cc_compound and cc_compound.inchi_key:
                    stoich[cc_compound] = count
            self._cc_reactions[reaction] = equilibrator_api.Reaction(stoich)
        return self._cc_reactions[reaction]

    def set_formation_delta_g(self, molecule: Molecule, dg_f: float):
        self._dg_f[molecule] = dg_f

    def formation_delta_g(self, molecule: Molecule) -> float:
        # equilibrator_api intentionally makes this harder to discourage using formation delta-G.
        # https://equilibrator.readthedocs.io/en/latest/equilibrator_examples.html#Using-formation-energies-to-calculate-reaction-energies
        if molecule in self._dg_f:
            return self._dg_f[molecule]

        cc_compound = self.cc_compound(molecule)
        if cc_compound and cc_compound.inchi_key:
            dgf_mu = self.cc.standard_dg_formation(cc_compound)[0]
            if dgf_mu is not None:
                legendre = cc_compound.transform(self.cc.p_h, self.cc.ionic_strength, self.cc.temperature, self.cc.p_mg)
                return dgf_mu + legendre.m_as("kJ/mol")
        # equilibrator_api does not return a value for e.g. protons, but documents that dg_f is 0 in this case.
        # We should come up with a more robust way of dealing with this. But settle for returning 0 for now.
        return 0

    def pkas(self, molecule: Molecule) -> List[float]:
        cc_compound = self.cc_compound(molecule)
        return cc_compound.dissociation_constants

    def reaction_delta_g(self, reaction: Reaction, concs: Optional[Mapping[Molecule, float]] = None) -> float:
        """Calculates ΔG° or ΔG for a reaction.
        
        Args:
            reaction: the reaction
            concs: Molar concentrations of reactants participating in the reaction

        Returns:
            If concs is None (the default), returns ΔG°. If concentrations are provided, returns ΔG = ΔG° + RT lnQ.
        """
        dg_r = self.cc.standard_dg_prime(self.cc_reaction(reaction)).value.m
        for reactant, count in reaction.stoichiometry.items():
            if reactant in self._dg_f:
                dg_r = dg_r + count * self._dg_f[reactant]
        if concs:
            ln_q = sum(np.log(concs.get(reactant, 1)) * count for reactant, count in reaction.stoichiometry.items())
            return dg_r + self.cc.RT.m * ln_q
        else:
            # Technically this is the same as RT * sum(np.log(concs.get(reactant, 1)) * count), which is the formal
            # definition of ΔG°. We just shortcut the computation.
            return dg_r
