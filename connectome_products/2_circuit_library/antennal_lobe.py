"""
Antennal Lobe Olfactory Processing Module
========================================

Bio-inspired model of the Drosophila Antennal Lobe (AL) primary olfactory center.
Simulates Odorant Receptor Neurons (ORNs), inhibitory Local Interneurons (LNs) providing
lateral gain control, and Projection Neurons (PNs) transmitting transformed odor representations.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


class AntennalLobeCircuit(nn.Module):
    """
    Antennal Lobe Glomerular Processing Circuit with LN Gain Control.

    Attributes:
        num_glomeruli (int): Number of distinct olfactory glomeruli (default 24).
        num_ln (int): Number of inhibitory local interneurons (default 12).
        tau_m (float): Membrane voltage decay time constant (ms).
        v_rest (float): Resting potential (mV).
        v_reset (float): Reset potential (mV).
        v_thresh (float): Firing threshold (mV).
        dt (float): Simulation timestep (ms).
    """

    def __init__(
        self,
        num_glomeruli: int = 24,
        num_ln: int = 12,
        gain_control_strength: float = 0.4,
        tau_m: float = 15.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -52.0,
        dt: float = 1.0,
    ):
        """
        Initializes AntennalLobeCircuit parameters and connectivity.

        Args:
            num_glomeruli (int): Count of olfactory channels/glomeruli. Defaults to 24.
            num_ln (int): Count of lateral inhibitory interneurons. Defaults to 12.
            gain_control_strength (float): Lateral inhibition strength. Defaults to 0.4.
            tau_m (float): LIF time constant (ms). Defaults to 15.0.
            v_rest (float): Resting membrane potential (mV). Defaults to -70.0.
            v_reset (float): Reset membrane potential (mV). Defaults to -70.0.
            v_thresh (float): Spike threshold (mV). Defaults to -52.0.
            dt (float): Simulation timestep (ms). Defaults to 1.0.
        """
        super().__init__()
        self.num_glomeruli = num_glomeruli
        self.num_ln = num_ln
        self.gain_control_strength = gain_control_strength
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.alpha = math.exp(-dt / tau_m)

        # ORN -> LN mapping (broad all-to-all connectivity)
        self.W_orn_ln = nn.Parameter(torch.ones(num_ln, num_glomeruli) / num_glomeruli, requires_grad=False)
        # LN -> PN inhibitory mapping (broad lateral inhibition matrix)
        self.W_ln_pn = nn.Parameter(torch.ones(num_glomeruli, num_ln) * gain_control_strength, requires_grad=False)

        # State buffers
        self.register_buffer("v_ln", torch.full((num_ln,), v_rest))
        self.register_buffer("v_pn", torch.full((num_glomeruli,), v_rest))
        self.register_buffer("ln_spikes", torch.zeros(num_ln))
        self.register_buffer("pn_spikes", torch.zeros(num_glomeruli))

    def reset_state(self) -> None:
        """Resets LN and PN membrane potentials and spike buffers."""
        self.v_ln.fill_(self.v_rest)
        self.v_pn.fill_(self.v_rest)
        self.ln_spikes.zero_()
        self.pn_spikes.zero_()

    def forward(self, orn_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executes single timestep forward update of Antennal Lobe circuit.

        Args:
            orn_input (Tensor): Input odor stimulus vector of shape (num_glomeruli,).

        Returns:
            Tuple[Tensor, Tensor, Dict[str, float]]:
                - pn_spikes (Tensor): Projection Neuron spike output vector of shape (num_glomeruli,).
                - ln_spikes (Tensor): Local Interneuron spike output vector of shape (num_ln,).
                - metrics (Dict[str, float]): Dynamic range and activity metrics.
        """
        # Drive LNs via total ORN input
        i_ln = torch.matmul(self.W_orn_ln, orn_input) * 20.0
        self.v_ln = self.v_rest + self.alpha * (self.v_ln - self.v_rest) + i_ln
        ln_spikes = (self.v_ln >= self.v_thresh).float()
        self.v_ln = torch.where(ln_spikes > 0, torch.tensor(self.v_reset, device=self.v_ln.device), self.v_ln)

        # Lateral inhibition current on PNs
        i_inh_pn = torch.matmul(self.W_ln_pn, ln_spikes) * 15.0
        i_exc_pn = orn_input * 25.0
        i_net_pn = torch.relu(i_exc_pn - i_inh_pn)

        # Drive PNs
        self.v_pn = self.v_rest + self.alpha * (self.v_pn - self.v_rest) + i_net_pn
        pn_spikes = (self.v_pn >= self.v_thresh).float()
        self.v_pn = torch.where(pn_spikes > 0, torch.tensor(self.v_reset, device=self.v_pn.device), self.v_pn)

        self.ln_spikes = ln_spikes
        self.pn_spikes = pn_spikes

        metrics = {
            "orn_input_sum": round(torch.sum(orn_input).item(), 4),
            "pn_active_ratio": round(torch.mean(pn_spikes).item(), 4),
            "ln_active_ratio": round(torch.mean(ln_spikes).item(), 4),
        }

        return pn_spikes, ln_spikes, metrics
