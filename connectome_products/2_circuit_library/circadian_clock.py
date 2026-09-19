"""
Circadian Pacemaker Neuropeptide Circuit Module
================================================

Bio-inspired model of Drosophila small Ventral Lateral Neurons (sLNv) circadian clock.
Simulates intracellular transcriptional feedback loops (PER/TIM) coupled with Pigment-Dispersing
Factor (PDF) neuropeptide signaling to synchronize morning activity rhythms.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


class CircadianClockCircuit(nn.Module):
    """
    Circadian Pacemaker & PDF Neuropeptide Signaling Network.

    Attributes:
        num_pacemakers (int): Number of sLNv pacemaker neurons.
        dt (float): Simulation step in hours (default 0.1 hour).
    """

    def __init__(
        self,
        num_pacemakers: int = 8,
        period_hours: float = 24.0,
        pdf_coupling_strength: float = 0.3,
        dt: float = 0.1,
    ):
        """
        Initializes CircadianClockCircuit parameters and state buffers.

        Args:
            num_pacemakers (int): Count of coupled sLNv neurons. Defaults to 8.
            period_hours (float): Natural circadian period in hours. Defaults to 24.0.
            pdf_coupling_strength (float): Inter-pacemaker PDF coupling weight. Defaults to 0.3.
            dt (float): Timestep in hours. Defaults to 0.1.
        """
        super().__init__()
        self.num_pacemakers = num_pacemakers
        self.period_hours = period_hours
        self.pdf_coupling_strength = pdf_coupling_strength
        self.dt = dt
        self.omega = (2.0 * math.pi) / period_hours

        # Phase angles of pacemakers
        init_phases = torch.linspace(0, 0.5, num_pacemakers)
        self.register_buffer("phase", init_phases)
        self.register_buffer("pdf_level", torch.zeros(num_pacemakers))
        self.register_buffer("clock_output", torch.zeros(num_pacemakers))

    def reset_state(self) -> None:
        """Resets circadian phase angles and PDF neuropeptide levels."""
        self.phase = torch.linspace(0, 0.5, self.num_pacemakers)
        self.pdf_level.zero_()
        self.clock_output.zero_()

    def forward(
        self, light_intensity: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executes single timestep forward update of circadian clock network.

        Args:
            light_intensity (float): External light cue (zeitgeber) input in range [0.0, 1.0].

        Returns:
            Tuple[Tensor, Tensor, Dict[str, float]]:
                - clock_output (Tensor): Sinusoidal morning activity drive [0.0, 1.0] of pacemakers.
                - pdf_level (Tensor): Released PDF neuropeptide concentration.
                - metrics (Dict[str, float]): Mean circadian phase and peak synchronization metrics.
        """
        # Kuramoto-style phase synchronization via PDF coupling
        sin_diffs = torch.sin(self.phase.unsqueeze(1) - self.phase.unsqueeze(0))
        coupling_force = (self.pdf_coupling_strength / self.num_pacemakers) * torch.sum(sin_diffs, dim=1)

        # Light pulse phase-resetting drive
        light_phase_reset = light_intensity * torch.sin(self.phase) * 0.2

        # Phase derivative d(phase)/dt = omega + coupling + light
        d_phase = (self.omega + coupling_force + light_phase_reset) * self.dt
        self.phase = (self.phase + d_phase) % (2.0 * math.pi)

        # Clock output peaks during subjective morning (phase ~ pi/2)
        self.clock_output = 0.5 * (1.0 + torch.sin(self.phase))
        self.pdf_level = 0.8 * self.pdf_level + 0.2 * self.clock_output

        mean_phase = torch.mean(self.phase).item()
        sync_order_parameter = torch.abs(torch.mean(torch.exp(1j * self.phase))).item()

        metrics = {
            "mean_circadian_phase_rad": round(mean_phase, 4),
            "synchronization_order_parameter": round(sync_order_parameter, 4),
            "mean_pdf_concentration": round(torch.mean(self.pdf_level).item(), 4),
            "subjective_time_hours": round((mean_phase / (2 * math.pi)) * self.period_hours, 2),
        }

        return self.clock_output, self.pdf_level, metrics
