"""
Optomotor Response Visual Stabilization Module
===============================================

Bio-inspired model of the Drosophila lobula plate tangent cells (LPTCs) optomotor circuit.
Integrates wide-field horizontal (HS) and vertical (VS) motion signals to generate corrective
torque commands for gaze and flight stabilization.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


class OptomotorResponseCircuit(nn.Module):
    """
    Optomotor Response Stabilization Circuit.

    Attributes:
        num_sensors (int): Number of wide-field motion input channels.
        tau_m (float): LIF membrane time constant (ms).
        v_rest (float): Resting membrane potential (mV).
        v_reset (float): Reset potential (mV).
        v_thresh (float): Spike threshold (mV).
        dt (float): Simulation timestep (ms).
    """

    def __init__(
        self,
        num_sensors: int = 16,
        tau_m: float = 12.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        dt: float = 1.0,
    ):
        """
        Initializes OptomotorResponseCircuit parameters and state buffers.

        Args:
            num_sensors (int): Number of horizontal/vertical motion sensing units. Defaults to 16.
            tau_m (float): LIF neuron time constant (ms). Defaults to 12.0.
            v_rest (float): Resting potential (mV). Defaults to -70.0.
            v_reset (float): Reset potential (mV). Defaults to -70.0.
            v_thresh (float): Firing threshold (mV). Defaults to -50.0.
            dt (float): Simulation timestep (ms). Defaults to 1.0.
        """
        super().__init__()
        self.num_sensors = num_sensors
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.alpha = math.exp(-dt / tau_m)

        # HS (Horizontal System) and VS (Vertical System) integrator neurons
        self.register_buffer("v_hs_left", torch.tensor(v_rest))
        self.register_buffer("v_hs_right", torch.tensor(v_rest))
        self.register_buffer("v_vs_pitch", torch.tensor(v_rest))
        self.register_buffer("spikes_hs_left", torch.tensor(0.0))
        self.register_buffer("spikes_hs_right", torch.tensor(0.0))
        self.register_buffer("spikes_vs_pitch", torch.tensor(0.0))

    def reset_state(self) -> None:
        """Resets membrane potentials and spike buffers."""
        self.v_hs_left.fill_(self.v_rest)
        self.v_hs_right.fill_(self.v_rest)
        self.v_vs_pitch.fill_(self.v_rest)
        self.spikes_hs_left.zero_()
        self.spikes_hs_right.zero_()
        self.spikes_vs_pitch.zero_()

    def forward(
        self, motion_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executes single timestep forward update of Optomotor stabilization circuit.

        Args:
            motion_vectors (Tensor): Wide-field visual motion vector tensor of shape (num_sensors,).
                Positive values indicate rightward motion, negative leftward.

        Returns:
            Tuple[Tensor, Tensor, Dict[str, float]]:
                - corrective_yaw (Tensor): Scalar corrective yaw torque output signal.
                - corrective_pitch (Tensor): Scalar corrective pitch torque output signal.
                - metrics (Dict[str, float]): Integrated motion metrics.
        """
        left_motion = torch.relu(-motion_vectors).sum()
        right_motion = torch.relu(motion_vectors).sum()

        i_hs_left = left_motion * 5.0
        i_hs_right = right_motion * 5.0
        i_vs = torch.abs(motion_vectors).mean() * 4.0

        # Update LIF neurons
        self.v_hs_left = self.v_rest + self.alpha * (self.v_hs_left - self.v_rest) + i_hs_left
        self.v_hs_right = self.v_rest + self.alpha * (self.v_hs_right - self.v_rest) + i_hs_right
        self.v_vs_pitch = self.v_rest + self.alpha * (self.v_vs_pitch - self.v_rest) + i_vs

        spikes_l = (self.v_hs_left >= self.v_thresh).float()
        spikes_r = (self.v_hs_right >= self.v_thresh).float()
        spikes_v = (self.v_vs_pitch >= self.v_thresh).float()

        if spikes_l > 0:
            self.v_hs_left.fill_(self.v_reset)
        if spikes_r > 0:
            self.v_hs_right.fill_(self.v_reset)
        if spikes_v > 0:
            self.v_vs_pitch.fill_(self.v_reset)

        self.spikes_hs_left = spikes_l
        self.spikes_hs_right = spikes_r
        self.spikes_vs_pitch = spikes_v

        # Corrective yaw steering torque: opposing optical flow rotation
        corrective_yaw = (right_motion - left_motion) * 0.1
        corrective_pitch = (self.v_vs_pitch - self.v_rest) * 0.05

        metrics = {
            "hs_left_v": round(self.v_hs_left.item(), 2),
            "hs_right_v": round(self.v_hs_right.item(), 2),
            "corrective_yaw_torque": round(corrective_yaw.item(), 4),
            "corrective_pitch_torque": round(corrective_pitch.item(), 4),
        }

        return corrective_yaw, corrective_pitch, metrics
