"""
Mushroom Body Kenyon Cell Dopamine-Modulated Plasticity Module
===============================================================

Bio-inspired model of the Drosophila Mushroom Body (MB) memory and learning circuit.
Encodes sparse olfactory/sensory patterns via Kenyon Cells (KCs) and applies
Dopaminergic Neuron (DAN) modulated long-term depression (LTD) at KC-to-MBON synapses
to encode associative memory (fear/reward conditioning).
"""

import math
import torch
import torch.nn as nn


class MushroomBodyLearningModule(nn.Module):
    """
    Mushroom Body Learning Module with Sparse KC Activation and DAN Dopamine Plasticity.
    
    Attributes:
        num_pn (int): Number of Projection Neurons (input sensory channels).
        num_kc (int): Number of Kenyon Cells (sparse expander layer).
        num_mbon (int): Number of Mushroom Body Output Neurons (valence decision channels).
        sparsity (float): Desired fraction of active KCs per pattern (~0.05).
        learning_rate (float): Rate of dopamine-modulated synaptic plasticity.
    """

    def __init__(
        self,
        num_pn: int = 50,
        num_kc: int = 2000,
        num_mbon: int = 4,
        sparsity: float = 0.05,
        learning_rate: float = 0.02,
        tau_m: float = 15.0,
        v_rest: float = -70.0,
        v_thresh: float = -55.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.num_pn = num_pn
        self.num_kc = num_kc
        self.num_mbon = num_mbon
        self.sparsity = sparsity
        self.learning_rate = learning_rate
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.dt = dt
        self.alpha = math.exp(-dt / tau_m)

        # Fixed sparse projection matrix PN -> KC (each KC receives ~6 random PN inputs)
        k_connections = 6
        W_pn_kc_init = torch.zeros(num_kc, num_pn)
        for i in range(num_kc):
            indices = torch.randperm(num_pn)[:k_connections]
            W_pn_kc_init[i, indices] = 1.0 / k_connections
        self.W_pn_kc = nn.Parameter(W_pn_kc_init, requires_grad=False)

        # Plastic weight matrix KC -> MBON (initialized to 1.0, driving approach)
        self.W_kc_mbon = nn.Parameter(torch.ones(num_mbon, num_kc) * 0.5, requires_grad=False)

        # State variables
        self.register_buffer("v_kc", torch.full((num_kc,), v_rest))
        self.register_buffer("kc_spikes", torch.zeros(num_kc))
        self.register_buffer("v_mbon", torch.full((num_mbon,), v_rest))
        self.register_buffer("mbon_spikes", torch.zeros(num_mbon))

    def reset_state(self):
        """Reset membrane potentials."""
        self.v_kc.fill_(self.v_rest)
        self.kc_spikes.zero_()
        self.v_mbon.fill_(self.v_rest)
        self.mbon_spikes.zero_()

    def forward(self, pn_input: torch.Tensor, dan_dopamine_signal: torch.Tensor = None):
        """
        Forward step of Mushroom Body circuit.
        
        Args:
            pn_input (Tensor): Sensory input from Projection Neurons of shape (num_pn,).
            dan_dopamine_signal (Tensor, optional): Dopamine punishment/reward signal of shape (num_mbon,).
            
        Returns:
            kc_spikes (Tensor): Sparse binary spike vector of Kenyon Cells (num_kc,).
            mbon_spikes (Tensor): Spike vector of Output Neurons (num_mbon,).
            kc_activity_ratio (float): Fraction of active Kenyon Cells.
        """
        # PN to KC synaptic drive
        i_kc = torch.matmul(self.W_pn_kc, pn_input)

        # Apply winner-take-all / global inhibition threshold to enforce KC sparsity (~5%)
        k_active = int(self.num_kc * self.sparsity)
        topk_vals, _ = torch.topk(i_kc, k_active)
        kc_threshold = topk_vals[-1] if len(topk_vals) > 0 else 1.0

        # LIF KC membrane update
        self.v_kc = self.v_rest + self.alpha * (self.v_kc - self.v_rest) + i_kc
        kc_spikes = (self.v_kc >= (self.v_rest + kc_threshold)).float()
        self.kc_spikes = kc_spikes
        self.v_kc = torch.where(kc_spikes > 0, torch.tensor(self.v_rest, device=self.v_kc.device), self.v_kc)

        # KC to MBON drive
        i_mbon = torch.matmul(self.W_kc_mbon, kc_spikes)
        self.v_mbon = self.v_rest + self.alpha * (self.v_mbon - self.v_rest) + i_mbon
        mbon_spikes = (self.v_mbon >= self.v_thresh).float()
        self.mbon_spikes = mbon_spikes
        self.v_mbon = torch.where(mbon_spikes > 0, torch.tensor(self.v_rest, device=self.v_mbon.device), self.v_mbon)

        # Dopamine-modulated synaptic plasticity update: anti-Hebbian LTD
        if dan_dopamine_signal is not None:
            self.apply_dopamine_plasticity(kc_spikes, dan_dopamine_signal)

        kc_activity_ratio = torch.mean(kc_spikes).item()
        return kc_spikes, mbon_spikes, kc_activity_ratio

    def apply_dopamine_plasticity(self, kc_spikes: torch.Tensor, dan_signal: torch.Tensor):
        """
        Dopamine-modulated STDP / LTD update:
        Delta W_ij = - learning_rate * KC_i * DAN_j
        Depresses active KC-MBON synapses when co-active with Dopamine release.
        """
        with torch.no_grad():
            dw = -self.learning_rate * torch.outer(dan_signal, kc_spikes)
            self.W_kc_mbon.add_(dw)
            self.W_kc_mbon.clamp_(min=0.0, max=1.0)
