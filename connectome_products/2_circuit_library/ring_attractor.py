"""
EPG Central Complex 360-degree Heading Direction Ring Attractor Circuit
========================================================================

Bio-inspired Leaky Integrate-and-Fire (LIF) spiking neural network model of the
Drosophila melanogaster Central Complex EPG (Ellipsoid Body - Protocerebral Bridge - Gall)
ring attractor. Maintains a stable bump of neural activity representing heading direction
and integrates angular velocity signals.
"""

import math
import torch
import torch.nn as nn
import numpy as np


class EPGRingAttractor(nn.Module):
    """
    EPG Ring Attractor Circuit for 360-degree Heading Direction Tracking.
    
    Attributes:
        num_wedges (int): Number of EPG wedges spanning 360 degrees (default 16).
        tau_m (float): Membrane time constant (ms).
        v_rest (float): Resting membrane potential (mV).
        v_reset (float): Reset membrane potential (mV).
        v_thresh (float): Spike threshold potential (mV).
        dt (float): Integration timestep (ms).
    """

    def __init__(
        self,
        num_wedges: int = 16,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        dt: float = 1.0,
        sigma_exc: float = 0.5,
        sigma_inh: float = 1.2,
        w_exc_gain: float = 4.0,
        w_inh_gain: float = 2.5
    ):
        super().__init__()
        self.num_wedges = num_wedges
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.alpha = math.exp(-dt / tau_m)

        # Angles corresponding to each wedge around the 360 degree circle
        self.angles = torch.linspace(0, 2 * math.pi, num_wedges + 1)[:-1]

        # Construct Mexican-Hat Recurrent Weight Matrix across circular topology
        self.W_rec = nn.Parameter(
            self._build_mexican_hat(sigma_exc, sigma_inh, w_exc_gain, w_inh_gain),
            requires_grad=False
        )

        # Asymmetric shift connections (PEN / P-EG neurons for angular velocity integration)
        self.W_shift_left = nn.Parameter(self._build_shift_matrix(-1), requires_grad=False)
        self.W_shift_right = nn.Parameter(self._build_shift_matrix(1), requires_grad=False)

        # State variables
        self.register_buffer("v_mem", torch.full((num_wedges,), v_rest))
        self.register_buffer("spikes", torch.zeros(num_wedges))

    def _build_mexican_hat(self, sigma_exc, sigma_inh, w_exc, w_inh):
        """Construct circular Mexican-hat distance-based weight matrix."""
        N = self.num_wedges
        W = torch.zeros(N, N)
        angles_np = np.linspace(0, 2 * np.pi, N, endpoint=False)
        
        for i in range(N):
            for j in range(N):
                # Minimal angular distance on circle
                diff = np.abs(angles_np[i] - angles_np[j])
                ang_dist = np.minimum(diff, 2 * np.pi - diff)
                
                exc = np.exp(-0.5 * (ang_dist / sigma_exc) ** 2)
                inh = np.exp(-0.5 * (ang_dist / sigma_inh) ** 2)
                W[i, j] = w_exc * exc - w_inh * inh
                
        # Self-excitation diagonal boost
        W = W - torch.diag(torch.diag(W)) + torch.eye(N) * (w_exc * 0.5)
        return W

    def _build_shift_matrix(self, shift: int):
        """Build circular shift matrix for velocity-driven bump movement."""
        N = self.num_wedges
        W = torch.zeros(N, N)
        for i in range(N):
            target = (i + shift) % N
            W[target, i] = 1.0
        return W

    def reset_state(self, initial_heading_rad: float = None):
        """Reset membrane potential and optionally inject initial heading bump."""
        self.v_mem.fill_(self.v_rest)
        self.spikes.zero_()
        
        if initial_heading_rad is not None:
            # Gaussian input bump centered around initial heading
            diff = torch.abs(self.angles - initial_heading_rad)
            ang_dist = torch.minimum(diff, 2 * math.pi - diff)
            bump = torch.exp(-0.5 * (ang_dist / 0.4) ** 2) * 15.0
            self.v_mem += bump

    def forward(self, angular_velocity: float = 0.0, external_cue: torch.Tensor = None):
        """
        Single timestep forward update of EPG Ring Attractor LIF circuit.
        
        Args:
            angular_velocity (float): Angular rotation rate (-rad/s to +rad/s).
            external_cue (torch.Tensor, optional): Visual/sensory cue tensor of shape (num_wedges,).
            
        Returns:
            spikes (Tensor): Binary spike vector of shape (num_wedges,).
            v_mem (Tensor): Membrane potentials of EPG neurons.
            heading_rad (float): Decoded population vector heading angle in radians [0, 2pi).
            heading_deg (float): Decoded population vector heading angle in degrees [0, 360).
        """
        # Recurrent synaptic current from previous spikes
        i_rec = torch.matmul(self.W_rec, self.spikes)

        # Angular velocity shift current (PEN left vs PEG right asymmetric activation)
        i_vel = torch.zeros_like(self.v_mem)
        if angular_velocity > 0:
            i_vel += angular_velocity * torch.matmul(self.W_shift_right, self.spikes)
        elif angular_velocity < 0:
            i_vel += abs(angular_velocity) * torch.matmul(self.W_shift_left, self.spikes)

        # Total synaptic input
        i_syn = i_rec + i_vel
        if external_cue is not None:
            i_syn += external_cue

        # LIF Membrane update
        self.v_mem = self.v_rest + self.alpha * (self.v_mem - self.v_rest) + i_syn

        # Spike generation with hard threshold
        spikes = (self.v_mem >= self.v_thresh).float()
        
        # Soft reset
        self.v_mem = torch.where(spikes > 0, torch.tensor(self.v_reset, device=self.v_mem.device), self.v_mem)
        self.spikes = spikes

        # Decode current heading from population activity
        heading_rad, heading_deg = self.decode_heading(self.v_mem - self.v_rest)

        return spikes, self.v_mem, heading_rad, heading_deg

    def decode_heading(self, activity: torch.Tensor):
        """Decode heading angle using circular population vector sum."""
        weights = torch.relu(activity)
        sum_weights = torch.sum(weights)
        
        if sum_weights < 1e-5:
            return 0.0, 0.0
            
        sin_sum = torch.sum(weights * torch.sin(self.angles))
        cos_sum = torch.sum(weights * torch.cos(self.angles))
        
        angle = math.atan2(sin_sum.item(), cos_sum.item())
        if angle < 0:
            angle += 2 * math.pi
            
        return angle, math.degrees(angle)
