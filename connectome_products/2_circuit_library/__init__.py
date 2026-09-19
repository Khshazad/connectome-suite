"""
Bio-Inspired PyTorch LIF Spiking Neural Network Circuit Library
================================================================

Exposes core neuromorphic Drosophila brain microcircuits:
- EPGRingAttractor: Central Complex 360-degree heading direction ring attractor.
- HassensteinReichardtEMD: Elementary Motion Detector visual motion correlator.
- GiantFiberEscapeCircuit: Giant Fiber rapid escape motor circuit.
- MushroomBodyLearningModule: Kenyon Cell dopamine-modulated synaptic plasticity.
"""

from .ring_attractor import EPGRingAttractor
from .motion_detector import HassensteinReichardtEMD
from .escape_circuit import GiantFiberEscapeCircuit
from .learning_circuit import MushroomBodyLearningModule

__all__ = [
    "EPGRingAttractor",
    "HassensteinReichardtEMD",
    "GiantFiberEscapeCircuit",
    "MushroomBodyLearningModule",
]
