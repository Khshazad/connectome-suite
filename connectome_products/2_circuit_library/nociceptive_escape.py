"""
Nociceptive Avoidance Reflex Circuit Module
============================================

Bio-inspired model of the Drosophila larval nociceptive rolling/bending reflex arc.
Integrates high-threshold thermal (TRPA1) and mechanical nociceptive sensory input through
multidendritic class IV (C4da) neurons to trigger downstream motor escape rolling responses.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Dict


class NociceptiveAvoidanceCircuit(nn.Module):
    """
    Nociceptive Avoidance Reflex Arc Circuit.

    Attributes:
        tau_noc (float): Nociceptors membrane time constant (ms).
        v_thresh_noc (float): High activation threshold (mV) for noxious stimuli.
        dt (float): Timestep (ms).
    """

    def __init__(
        self,
        tau_noc: float = 5.0,
        tau_motor: float = 8.0,
        v_rest: float = -70.0,
        v_reset: float = -70.0,
        v_thresh_noc: float = -42.0,
        v_thresh_roll: float = -50.0,
        dt: float = 1.0,
    ):
        """
        Initializes NociceptiveAvoidanceCircuit parameters.

        Args:
            tau_noc (float): C4da nociceptor neuron time constant (ms). Defaults to 5.0.
            tau_motor (float): Downstream motor interneuron time constant (ms). Defaults to 8.0.
            v_rest (float): Resting potential (mV). Defaults to -70.0.
            v_reset (float): Reset potential (mV). Defaults to -70.0.
            v_thresh_noc (float): High threshold for noxious activation (mV). Defaults to -42.0.
            v_thresh_roll (float): Rolling reflex threshold (mV). Defaults to -50.0.
            dt (float): Simulation timestep (ms). Defaults to 1.0.
        """
        super().__init__()
        self.tau_noc = tau_noc
        self.tau_motor = tau_motor
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.v_thresh_noc = v_thresh_noc
        self.v_thresh_roll = v_thresh_roll
        self.dt = dt

        self.alpha_noc = math.exp(-dt / tau_noc)
        self.alpha_motor = math.exp(-dt / tau_motor)

        # State buffers
        self.register_buffer("v_c4da", torch.tensor(v_rest))
        self.register_buffer("v_basin", torch.tensor(v_rest))
        self.register_buffer("v_roll_motor", torch.tensor(v_rest))

    def reset_state(self) -> None:
        """Resets all membrane potentials."""
        self.v_c4da.fill_(self.v_rest)
        self.v_basin.fill_(self.v_rest)
        self.v_roll_motor.fill_(self.v_rest)

    def forward(
        self, thermal_stimulus: float = 0.0, mechanical_stimulus: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Executes single timestep forward update of nociceptive reflex circuit.

        Args:
            thermal_stimulus (float): Noxious heat intensity (degrees C above 38C).
            mechanical_stimulus (float): High mechanical force impulse (mN).

        Returns:
            Tuple[Tensor, Tensor, Dict[str, float]]:
                - roll_spike (Tensor): Spike activation of rolling motor neuron.
                - bend_spike (Tensor): Spike activation of bending motor neuron.
                - metrics (Dict[str, float]): Threat level and activation status.
        """
        # C4da nociceptive integration: high threshold activation
        i_thermal = max(0.0, thermal_stimulus) * 6.0
        i_mech = max(0.0, mechanical_stimulus) * 8.0
        i_noxious = i_thermal + i_mech

        # Update C4da sensor LIF
        self.v_c4da = self.v_rest + self.alpha_noc * (self.v_c4da - self.v_rest) + i_noxious
        c4da_spike = (self.v_c4da >= self.v_thresh_noc).float()
        if c4da_spike > 0:
            self.v_c4da.fill_(self.v_reset)

        # Drive downstream Basin interneurons
        i_basin = c4da_spike * 30.0
        self.v_basin = self.v_rest + self.alpha_motor * (self.v_basin - self.v_rest) + i_basin
        basin_spike = (self.v_basin >= self.v_thresh_roll).float()
        if basin_spike > 0:
            self.v_basin.fill_(self.v_reset)

        # Drive rolling reflex motor neurons
        i_roll = basin_spike * 40.0
        self.v_roll_motor = self.v_rest + self.alpha_motor * (self.v_roll_motor - self.v_rest) + i_roll
        roll_spike = (self.v_roll_motor >= self.v_thresh_roll).float()
        if roll_spike > 0:
            self.v_roll_motor.fill_(self.v_reset)

        bend_spike = (basin_spike > 0 and roll_spike == 0).float()

        metrics = {
            "c4da_membrane_v": round(self.v_c4da.item(), 2),
            "basin_membrane_v": round(self.v_basin.item(), 2),
            "roll_motor_active": roll_spike.item() > 0,
            "reflex_state": "ROLLING_ESCAPE" if roll_spike.item() > 0 else ("BENDING" if bend_spike.item() > 0 else "QUIESCENT"),
        }

        return roll_spike, bend_spike, metrics
