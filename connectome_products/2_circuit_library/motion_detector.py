"""
Elementary Motion Detector (EMD) Hassenstein-Reichardt Correlator Circuit
========================================================================

Bio-inspired Elementary Motion Detector (EMD) implementation based on the 
classic Hassenstein-Reichardt model found in the Drosophila optic lobe (T4/T5 cells).
Correlates temporally delayed signals from adjacent visual channels to detect preferred
directional motion across 1D visual arrays or 2D image inputs.
"""

import math
import torch
import torch.nn as nn


class HassensteinReichardtEMD(nn.Module):
    """
    Hassenstein-Reichardt Elementary Motion Detector (EMD) with LIF Spiking Neurons.
    
    Attributes:
        num_channels (int): Number of adjacent visual receptor channels.
        tau_delay (float): Delay low-pass filter time constant (ms).
        tau_m (float): LIF neuron membrane time constant (ms).
        dt (float): Integration timestep (ms).
    """

    def __init__(
        self,
        num_channels: int = 16,
        tau_delay: float = 15.0,
        tau_m: float = 10.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -52.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.num_channels = num_channels
        self.tau_delay = tau_delay
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt

        self.alpha_delay = math.exp(-dt / tau_delay)
        self.alpha_m = math.exp(-dt / tau_m)

        # State buffers
        self.register_buffer("delayed_state", torch.zeros(num_channels))
        self.register_buffer("v_mem_left", torch.full((num_channels - 1,), v_rest))
        self.register_buffer("v_mem_right", torch.full((num_channels - 1,), v_rest))

    def reset_state(self):
        """Reset internal delay filters and neuron membrane potentials."""
        self.delayed_state.zero_()
        self.v_mem_left.fill_(self.v_rest)
        self.v_mem_right.fill_(self.v_rest)

    def forward(self, input_luminance: torch.Tensor):
        """
        Process single timestep visual input luminance vector.
        
        Args:
            input_luminance (Tensor): 1D visual receptor signals of shape (num_channels,).
            
        Returns:
            emd_net (Tensor): Net EMD directional motion signals of shape (num_channels - 1,).
            spikes_left (Tensor): Leftward motion detection spikes.
            spikes_right (Tensor): Rightward motion detection spikes.
            net_motion (float): Integrated scalar motion across visual field (-left / +right).
        """
        # Low-pass filter delay arm: S_delayed[t] = alpha * S_delayed[t-1] + (1 - alpha) * R[t]
        prev_delayed = self.delayed_state.clone()
        self.delayed_state = self.alpha_delay * self.delayed_state + (1.0 - self.alpha_delay) * input_luminance

        # Correlate adjacent channels
        # Left channel R[i], Right channel R[i+1]
        r_curr_left = input_luminance[:-1]
        r_curr_right = input_luminance[1:]
        r_del_left = prev_delayed[:-1]
        r_del_right = prev_delayed[1:]

        # Hassenstein-Reichardt multiplication
        # Rightward motion: delayed left correlated with current right
        corr_rightward = r_del_left * r_curr_right
        # Leftward motion: current left correlated with delayed right
        corr_leftward = r_curr_left * r_del_right

        # Net motion signal (positive = rightward, negative = leftward)
        emd_net = corr_rightward - corr_leftward

        # Drive LIF spiking motion sensitive neurons
        i_right = torch.relu(emd_net) * 20.0
        i_left = torch.relu(-emd_net) * 20.0

        self.v_mem_right = self.v_rest + self.alpha_m * (self.v_mem_right - self.v_rest) + i_right
        self.v_mem_left = self.v_rest + self.alpha_m * (self.v_mem_left - self.v_rest) + i_left

        spikes_right = (self.v_mem_right >= self.v_thresh).float()
        spikes_left = (self.v_mem_left >= self.v_thresh).float()

        self.v_mem_right = torch.where(spikes_right > 0, torch.tensor(self.v_reset, device=self.v_mem_right.device), self.v_mem_right)
        self.v_mem_left = torch.where(spikes_left > 0, torch.tensor(self.v_reset, device=self.v_mem_left.device), self.v_mem_left)

        net_motion = torch.sum(emd_net).item()

        return emd_net, spikes_left, spikes_right, net_motion
