"""
Spiking Neuron Models Module
============================

Provides PyTorch nn.Module implementations of fundamental bio-inspired spiking neuron models:
1. LIFNeuron: Leaky Integrate-and-Fire model with exponential membrane decay.
2. IzhikevichNeuron: Biologically plausible 2D dynamical system supporting diverse firing modes
   (regular spiking, fast spiking, bursting, chattering).
3. AdExNeuron: Adaptive Exponential Integrate-and-Fire model with exponential spike initiation threshold
   and subthreshold/spike-triggered adaptation currents.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Spiking Neuron.

    Models passive RC membrane voltage dynamics:
        tau_m * dV/dt = -(V - V_rest) + R * I_syn

    Attributes:
        num_neurons (int): Number of neuron instances in layer.
        tau_m (float): Membrane time constant in milliseconds.
        v_rest (float): Resting membrane potential in mV.
        v_reset (float): Reset membrane potential in mV after spike.
        v_thresh (float): Firing threshold potential in mV.
        dt (float): Integration timestep in milliseconds.
    """

    def __init__(
        self,
        num_neurons: int = 1,
        tau_m: float = 20.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh: float = -50.0,
        dt: float = 1.0,
    ):
        """
        Initializes LIFNeuron parameters and state variables.

        Args:
            num_neurons (int): Number of parallel neuron units. Defaults to 1.
            tau_m (float): Membrane decay time constant (ms). Defaults to 20.0.
            v_rest (float): Resting potential (mV). Defaults to -70.0.
            v_reset (float): Reset potential (mV). Defaults to -70.0.
            v_thresh (float): Spike threshold potential (mV). Defaults to -50.0.
            dt (float): Simulation timestep (ms). Defaults to 1.0.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh = v_thresh
        self.dt = dt
        self.alpha = math.exp(-dt / tau_m)

        self.register_buffer("v_mem", torch.full((num_neurons,), v_rest))
        self.register_buffer("spikes", torch.zeros(num_neurons))

    def reset_state(self, v_init: Optional[float] = None) -> None:
        """
        Resets membrane potential and spike state.

        Args:
            v_init (float, optional): Initial membrane potential. Defaults to v_rest.
        """
        v_val = v_init if v_init is not None else self.v_rest
        self.v_mem.fill_(v_val)
        self.spikes.zero_()

    def forward(self, i_syn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes single timestep forward update.

        Args:
            i_syn (Tensor): Synaptic current input tensor of shape (num_neurons,).

        Returns:
            Tuple[Tensor, Tensor]:
                - spikes (Tensor): Binary spike output (1.0 = spike, 0.0 = no spike).
                - v_mem (Tensor): Updated membrane potential (mV).
        """
        self.v_mem = self.v_rest + self.alpha * (self.v_mem - self.v_rest) + i_syn
        spikes = (self.v_mem >= self.v_thresh).float()
        self.v_mem = torch.where(spikes > 0, torch.tensor(self.v_reset, device=self.v_mem.device), self.v_mem)
        self.spikes = spikes
        return spikes, self.v_mem


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich Spiking Neuron Model.

    Calculates fast 2D membrane potential v and recovery variable u dynamics:
        dv/dt = 0.04*v^2 + 5*v + 140 - u + I
        du/dt = a * (b*v - u)
        if v >= v_peak (30 mV):
            v <- c
            u <- u + d

    Attributes:
        num_neurons (int): Number of neuron instances.
        a (float): Time scale of recovery variable u.
        b (float): Sensitivity of recovery u to subthreshold fluctuations of v.
        c (float): After-spike reset value of v (mV).
        d (float): After-spike reset increment of u.
        v_peak (float): Peak threshold potential for spike cutoff (mV).
        dt (float): Simulation timestep (ms).
    """

    def __init__(
        self,
        num_neurons: int = 1,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_peak: float = 30.0,
        dt: float = 0.5,
    ):
        """
        Initializes IzhikevichNeuron parameters and state buffers.

        Args:
            num_neurons (int): Number of parallel neurons. Defaults to 1.
            a (float): Recovery time scale parameter. Defaults to 0.02 (Regular Spiking).
            b (float): Recovery sensitivity parameter. Defaults to 0.2.
            c (float): Voltage reset value (mV). Defaults to -65.0.
            d (float): Recovery variable increment. Defaults to 8.0.
            v_peak (float): Peak voltage threshold for spike fire (mV). Defaults to 30.0.
            dt (float): Timestep integration resolution (ms). Defaults to 0.5.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_peak = v_peak
        self.dt = dt

        self.register_buffer("v_mem", torch.full((num_neurons,), -65.0))
        self.register_buffer("u_rec", torch.full((num_neurons,), b * -65.0))
        self.register_buffer("spikes", torch.zeros(num_neurons))

    def reset_state(self, v_init: float = -65.0) -> None:
        """
        Resets voltage v and recovery variable u.

        Args:
            v_init (float): Initial membrane potential in mV. Defaults to -65.0.
        """
        self.v_mem.fill_(v_init)
        self.u_rec.fill_(self.b * v_init)
        self.spikes.zero_()

    def forward(self, i_syn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes single step numerical integration using Euler method.

        Args:
            i_syn (Tensor): Input current tensor of shape (num_neurons,).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - spikes (Tensor): Binary spike output tensor.
                - v_mem (Tensor): Membrane voltage v.
                - u_rec (Tensor): Recovery variable u.
        """
        v = self.v_mem
        u = self.u_rec

        # Sub-step integration for numerical stability
        steps = max(1, int(round(1.0 / self.dt)))
        sub_dt = self.dt

        for _ in range(steps):
            dv = (0.04 * v**2 + 5.0 * v + 140.0 - u + i_syn) * sub_dt
            du = (self.a * (self.b * v - u)) * sub_dt
            v = v + dv
            u = u + du

        spikes = (v >= self.v_peak).float()

        # Reset condition
        v = torch.where(spikes > 0, torch.tensor(self.c, device=v.device), v)
        u = torch.where(spikes > 0, u + self.d, u)

        self.v_mem = v
        self.u_rec = u
        self.spikes = spikes

        return spikes, self.v_mem, self.u_rec


class AdExNeuron(nn.Module):
    """
    Adaptive Exponential Integrate-and-Fire (AdEx) Spiking Neuron.

    Models non-linear exponential spike initiation and adaptation current w:
        C * dV/dt = -g_L * (V - E_L) + g_L * delta_T * exp((V - V_T) / delta_T) - w + I
        tau_w * dw/dt = a * (V - E_L) - w
        if V >= V_peak:
            V <- V_reset
            w <- w + b

    Attributes:
        num_neurons (int): Number of parallel neurons.
        C (float): Membrane capacitance (pF).
        g_L (float): Leak conductance (nS).
        e_l (float): Resting leak reversal potential (mV).
        v_reset (float): Reset potential (mV).
        v_t (float): Exponential threshold potential (mV).
        delta_t (float): Slope factor (mV).
        a (float): Subthreshold adaptation parameter (nS).
        tau_w (float): Adaptation time constant (ms).
        b (float): Spike-triggered adaptation increment (pA).
        v_peak (float): Peak threshold for spike (mV).
        dt (float): Integration step (ms).
    """

    def __init__(
        self,
        num_neurons: int = 1,
        C: float = 200.0,
        g_L: float = 10.0,
        e_l: float = -70.0,
        v_reset: float = -58.0,
        v_t: float = -50.0,
        delta_t: float = 2.0,
        a: float = 2.0,
        tau_w: float = 30.0,
        b: float = 60.0,
        v_peak: float = 0.0,
        dt: float = 0.5,
    ):
        """
        Initializes AdExNeuron parameters.

        Args:
            num_neurons (int): Count of parallel neurons. Defaults to 1.
            C (float): Membrane capacitance (pF). Defaults to 200.0.
            g_L (float): Leak conductance (nS). Defaults to 10.0.
            e_l (float): Leak reversal potential (mV). Defaults to -70.0.
            v_reset (float): Reset voltage (mV). Defaults to -58.0.
            v_t (float): Spike threshold voltage (mV). Defaults to -50.0.
            delta_t (float): Exponential slope factor (mV). Defaults to 2.0.
            a (float): Subthreshold adaptation (nS). Defaults to 2.0.
            tau_w (float): Adaptation time constant (ms). Defaults to 30.0.
            b (float): Spike-triggered adaptation (pA). Defaults to 60.0.
            v_peak (float): Peak spike threshold (mV). Defaults to 0.0.
            dt (float): Timestep (ms). Defaults to 0.5.
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.C = C
        self.g_L = g_L
        self.e_l = e_l
        self.v_reset = v_reset
        self.v_t = v_t
        self.delta_t = delta_t
        self.a = a
        self.tau_w = tau_w
        self.b = b
        self.v_peak = v_peak
        self.dt = dt

        self.register_buffer("v_mem", torch.full((num_neurons,), e_l))
        self.register_buffer("w_adapt", torch.zeros(num_neurons))
        self.register_buffer("spikes", torch.zeros(num_neurons))

    def reset_state(self, v_init: Optional[float] = None) -> None:
        """
        Resets membrane voltage V and adaptation current w.

        Args:
            v_init (float, optional): Initial voltage. Defaults to e_l.
        """
        v_val = v_init if v_init is not None else self.e_l
        self.v_mem.fill_(v_val)
        self.w_adapt.zero_()
        self.spikes.zero_()

    def forward(self, i_syn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes AdEx Euler forward update.

        Args:
            i_syn (Tensor): Input current tensor (pA) of shape (num_neurons,).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - spikes (Tensor): Binary spike output tensor.
                - v_mem (Tensor): Membrane voltage V (mV).
                - w_adapt (Tensor): Adaptation current w (pA).
        """
        v = self.v_mem
        w = self.w_adapt
        dt = self.dt

        # Exponential spike component clipped to prevent numerical blowup
        exp_term = torch.exp(torch.clamp((v - self.v_t) / self.delta_t, max=10.0))
        dv = ((-self.g_L * (v - self.e_l) + self.g_L * self.delta_t * exp_term - w + i_syn) / self.C) * dt
        dw = ((self.a * (v - self.e_l) - w) / self.tau_w) * dt

        v = v + dv
        w = w + dw

        spikes = (v >= self.v_peak).float()

        v = torch.where(spikes > 0, torch.tensor(self.v_reset, device=v.device), v)
        w = torch.where(spikes > 0, w + self.b, w)

        self.v_mem = v
        self.w_adapt = w
        self.spikes = spikes

        return spikes, self.v_mem, self.w_adapt
