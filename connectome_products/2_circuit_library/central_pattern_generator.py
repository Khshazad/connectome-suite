"""
Central Pattern Generator (CPG) Rhythmic Motor Pattern Module
============================================================

Bio-inspired Central Pattern Generator (CPG) circuit model based on Drosophila thoracic
ganglion motor networks. Uses coupled half-center LIF oscillators with mutual reciprocal
inhibition to generate rhythmic alternating flexor/extensor leg and elevator/depressor wing
locomotor patterns.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


class CPGMotorGenerator(nn.Module):
    """
    Central Pattern Generator (CPG) for Rhythmic Motor Control.

    Attributes:
        num_limbs (int): Number of coupled motor pairs (e.g. 6 legs or 2 wings).
        tau_m (float): LIF membrane time constant (ms).
        w_inh (float): Reciprocal inhibitory weight strength between antagonist pools.
        w_adapt (float): Spike frequency adaptation conductance strength.
        v_rest (float): Resting membrane potential (mV).
        v_reset (float): Reset potential (mV).
        v_thresh (float): Spike threshold (mV).
        dt (float): Integration timestep (ms).
    """

    def __init__(
        self,
        num_limbs: int = 6,
        tau_m: float = 10.0,
        tau_adapt: float = 50.0,
        w_inh: float = 8.0,
        w_adapt: float = 5.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        dt: float = 1.0,
    ):
        """
        Initializes CPGMotorGenerator circuit parameters and state buffers.

        Args:
            num_limbs (int): Number of antagonist pairs (legs/wings). Defaults to 6.
            tau_m (float): LIF neuron time constant (ms). Defaults to 10.0.
            tau_adapt (float): Adaptation decay time constant (ms). Defaults to 50.0.
            w_inh (float): Reciprocal inhibition weight between flexor and extensor. Defaults to 8.0.
            w_adapt (float): Self-adaptation inhibition weight. Defaults to 5.0.
            v_rest (float): Resting membrane potential (mV). Defaults to -70.0.
            v_reset (float): Reset membrane potential (mV). Defaults to -70.0.
            v_thresh (float): Spike threshold (mV). Defaults to -50.0.
            dt (float): Simulation timestep (ms). Defaults to 1.0.
        """
        super().__init__()
        self.num_limbs = num_limbs
        self.tau_m = tau_m
        self.tau_adapt = tau_adapt
        self.w_inh = w_inh
        self.w_adapt = w_adapt
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt

        self.alpha_m = math.exp(-dt / tau_m)
        self.alpha_adapt = math.exp(-dt / tau_adapt)

        # State buffers for Flexor (Pool A) and Extensor (Pool B) motor pools
        self.register_buffer("v_flexor", torch.full((num_limbs,), v_rest))
        self.register_buffer("v_extensor", torch.full((num_limbs,), v_rest))
        self.register_buffer("adapt_flexor", torch.zeros(num_limbs))
        self.register_buffer("adapt_extensor", torch.zeros(num_limbs))
        self.register_buffer("spikes_flexor", torch.zeros(num_limbs))
        self.register_buffer("spikes_extensor", torch.zeros(num_limbs))

    def reset_state(self) -> None:
        """Resets all membrane potentials, adaptation currents, and spike buffers."""
        self.v_flexor.fill_(self.v_rest)
        self.v_extensor.fill_(self.v_rest)
        self.adapt_flexor.zero_()
        self.adapt_extensor.zero_()
        self.spikes_flexor.zero_()
        self.spikes_extensor.zero_()

    def forward(
        self, tonic_drive: float = 15.0, interlimb_coupling: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executes single timestep forward update of CPG motor generator.

        Args:
            tonic_drive (float): Descending excitatory drive from brain to drive oscillation.
            interlimb_coupling (Tensor, optional): Coupling tensor between limbs of shape (num_limbs, num_limbs).

        Returns:
            Tuple[Tensor, Tensor, Dict[str, float]]:
                - flexor_spikes (Tensor): Spike vector of flexor motor neurons (num_limbs,).
                - extensor_spikes (Tensor): Spike vector of extensor motor neurons (num_limbs,).
                - metrics (Dict[str, float]): Dictionary containing oscillation phase and frequency metrics.
        """
        # Reciprocal inhibition inputs
        i_inh_flexor = self.w_inh * self.spikes_extensor + self.w_adapt * self.adapt_flexor
        i_inh_extensor = self.w_inh * self.spikes_flexor + self.w_adapt * self.adapt_extensor

        # Total currents
        i_flexor = tonic_drive - i_inh_flexor
        i_extensor = tonic_drive - i_inh_extensor

        # Interlimb phase coupling (tripod gait coupling)
        if interlimb_coupling is not None:
            i_flexor += torch.matmul(interlimb_coupling, self.spikes_flexor)
            i_extensor += torch.matmul(interlimb_coupling, self.spikes_extensor)

        # Update LIF membrane potentials
        self.v_flexor = self.v_rest + self.alpha_m * (self.v_flexor - self.v_rest) + i_flexor
        self.v_extensor = self.v_rest + self.alpha_m * (self.v_extensor - self.v_rest) + i_extensor

        # Generate spikes
        spikes_f = (self.v_flexor >= self.v_thresh).float()
        spikes_e = (self.v_extensor >= self.v_thresh).float()

        # Update adaptation variables
        self.adapt_flexor = self.alpha_adapt * self.adapt_flexor + spikes_f
        self.adapt_extensor = self.alpha_adapt * self.adapt_extensor + spikes_e

        # Reset membrane potentials
        self.v_flexor = torch.where(spikes_f > 0, torch.tensor(self.v_reset, device=self.v_flexor.device), self.v_flexor)
        self.v_extensor = torch.where(spikes_e > 0, torch.tensor(self.v_reset, device=self.v_extensor.device), self.v_extensor)

        self.spikes_flexor = spikes_f
        self.spikes_extensor = spikes_e

        # Phase estimation (Flexor vs Extensor activity ratio)
        mean_f = torch.mean(spikes_f).item()
        mean_e = torch.mean(spikes_e).item()
        duty_cycle = mean_f / (mean_f + mean_e + 1e-6)

        metrics = {
            "flexor_activity": round(mean_f, 4),
            "extensor_activity": round(mean_e, 4),
            "duty_cycle": round(duty_cycle, 4),
        }

        return spikes_f, spikes_e, metrics
