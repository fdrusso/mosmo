from typing import Iterable, Mapping, Optional, Union

from vivarium.core.process import Process
from vivarium.core.types import State, Update

from kb import kb
from model.reaction_network import ReactionNetwork
from sim.fba_gd import FbaGd, ProductionObjective

KB = kb.configure_kb()


class FbaProcess(Process):
    """A Vivarium Process simulating a ReactionNetwork via gradient-descent FBA."""

    def __init__(
            self,
            pathways: Iterable[str],
            reactions: Iterable[str],
            boundaries: Mapping[str, Optional[float]],
            gain: float = 0.5):
        super().__init__()

        self.gain = gain

        # Build up a non-redundant set of reactions to include in the network.
        steps = set()
        for pathway_id in (pathways or []):
            pathway = KB.get(KB.pathways, pathway_id)
            steps.update(pathway.steps)
        for reaction_id in (reactions or []):
            steps.add(KB.get(KB.reactions, reaction_id))
        self.network = ReactionNetwork(steps)

        # Resolve boundary metabolites and selected targets.
        self.boundaries = set()
        self.targets = {}
        for met_id, target in boundaries.items():
            met = KB.get(KB.compounds, met_id)
            self.boundaries.add(met)
            if target is not None:
                self.targets[met] = target

        # Set up the FBA problem. Everything not declared as a boundary is an intermediate.
        self.intermediates = [met for met in self.network.reactants() if met not in self.boundaries]
        self.fba = FbaGd(self.network, self.intermediates, {
            'targets': ProductionObjective(self.network, {met: 0.0 for met, target in self.targets.items()})
        })

    def ports_schema(self):
        return {
            'metabolites': {
                met.id: {'_default': 0.0, '_emit': True} for met in self.boundaries
            },
            'fluxes': {
                rxn.id: {'_updater': 'set', '_emit': True} for rxn in self.network.reactions()
            },
        }

    def next_update(self, time_step: Union[float, int], states: State) -> Update:
        # Update the FBA problem with target rates based on current boundary metabolite pools.
        targets = {}
        for met, target in self.targets.items():
            # Current and target values are concentrations, in mM.
            current = states['metabolites'][met.id]
            # Target rates depend on displacement, invariant to time_step, controlled by a gain parameter.
            targets[met] = (target - current) * self.gain
        self.fba.update_params({'targets': targets})

        # Solve the problem and return updates
        soln = self.fba.solve()
        rates = self.network.reactant_values(soln.dmdt)
        velocities = self.network.reaction_values(soln.velocities)
        return {
            'metabolites': {met.id: rate * time_step for met, rate in rates.items()},
            'fluxes': {rxn.id: velocity for rxn, velocity in velocities.items()},
        }
