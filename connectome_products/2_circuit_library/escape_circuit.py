"""
Giant Fiber (GF) Escape Motor Circuit
=====================================

Bio-inspired model of the Drosophila Giant Fiber escape system.
Integrates visual looming threat signals (expansion rate r/v) and mechanosensory cues,
activating a high-threshold Giant Fiber neuron that triggers rapid leg jump (TTM)
and wing elevation (DLM) escape motor responses within milliseconds.
"""

import math
import torch
import torch.nn as nn


class GiantFiberEscapeCircuit(nn.Module):
    """
    Giant Fiber (GF) Escape Motor Circuit with Looming Threat & Spike Thresholding.
    
    Attributes:
        tau_gf (float): Giant fiber membrane time constant (ms, fast ~3.0 ms).
        v_thresh_gf (float): High spike threshold for explosive escape trigger (-45 mV).
    """

    def __init__(
        self,
        tau_gf: float = 3.0,
        tau_motor: float = 5.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh_gf: float = -45.0,
        v_thresh_motor: float = -55.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.tau_gf = tau_gf
        self.tau_motor = tau_motor
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh_gf = v_thresh_gf
        self.v_thresh_motor = v_thresh_motor
        self.dt = dt

        self.alpha_gf = math.exp(-dt / tau_gf)
        self.alpha_motor = math.exp(-dt / tau_motor)

        # State variables
        self.register_buffer("v_gf", torch.tensor(v_rest))
        self.register_buffer("v_ttm", torch.tensor(v_rest))  # Tergotrochanteral motor neuron (Leg jump)
        self.register_buffer("v_dlm", torch.tensor(v_rest))  # Dorsolongitudinal motor neuron (Wing flap)
        self.register_buffer("escape_triggered", torch.tensor(0.0))

    def reset_state(self):
        """Reset membrane potentials and escape status."""
        self.v_gf.fill_(self.v_rest)
        self.v_ttm.fill_(self.v_rest)
        self.v_dlm.fill_(self.v_rest)
        self.escape_triggered.zero_()

    def forward(self, angular_expansion_rate: float, mechanosensory_cue: float = 0.0):
        """
        Process visual looming threat expansion rate (rad/s) and mechanosensory input.
        
        Args:
            angular_expansion_rate (float): Visual looming expansion rate (theta_dot).
            mechanosensory_cue (float): Somatosensory touch / wind impulse signal.
            
        Returns:
            gf_spike (Tensor): Binary spike signal of Giant Fiber neuron.
            ttm_jump (Tensor): Spike activation of Tergotrochanteral motor neuron (Jump).
            dlm_wings (Tensor): Spike activation of Dorsolongitudinal motor neuron (Wings).
            threat_level (float): Integrated membrane potential of Giant Fiber.
        """
        # Non-linear looming integration: r/v exponential expansion factor
        loom_drive = math.pow(max(0.0, angular_expansion_rate), 1.8) * 15.0
        mech_drive = max(0.0, mechanosensory_cue) * 25.0
        i_total = loom_drive + mech_drive

        # Update Giant Fiber LIF membrane potential
        self.v_gf = self.v_rest + self.alpha_gf * (self.v_gf - self.v_rest) + i_total

        # Spike generation
        gf_spike = (self.v_gf >= self.v_thresh_gf).float()

        if gf_spike > 0:
            self.v_gf.fill_(self.v_reset)
            self.escape_triggered.fill_(1.0)

        # GF drives downstream motor interneurons TTM and DLM via electrical/chemical synapses
        i_gf_syn = gf_spike * 35.0

        self.v_ttm = self.v_rest + self.alpha_motor * (self.v_ttm - self.v_rest) + i_gf_syn
        self.v_dlm = self.v_rest + self.alpha_motor * (self.v_dlm - self.v_rest) + i_gf_syn * 0.8

        ttm_jump = (self.v_ttm >= self.v_thresh_motor).float()
        dlm_wings = (self.v_dlm >= self.v_thresh_motor).float()

        if ttm_jump > 0:
            self.v_ttm.fill_(self.v_reset)
        if dlm_wings > 0:
            self.v_dlm.fill_(self.v_reset)

        return gf_spike, ttm_jump, dlm_wings, self.v_gf.item()
