"""
Bio-Inspired PyTorch Spiking Neural Network Circuit Library SDK
================================================================

Full 10-circuit Drosophila melanogaster brain microcircuit SDK:
1. EPGRingAttractor: Central Complex 360-degree heading direction ring attractor model.
2. HassensteinReichardtEMD: Elementary Motion Detector visual motion correlator.
3. GiantFiberEscapeCircuit: Giant Fiber rapid looming escape motor circuit.
4. MushroomBodyLearningModule: Kenyon Cell dopamine-modulated synaptic plasticity.
5. CPGMotorGenerator: Central Pattern Generator rhythmic leg/wing motor pattern generator.
6. LIFNeuron, IzhikevichNeuron, AdExNeuron: Fundamental spiking neuron models.
7. AntennalLobeCircuit: Antennal Lobe gain control & lateral inhibition olfactory module.
8. OptomotorResponseCircuit: Visual wide-field optomotor gaze stabilization circuit.
9. CircadianClockCircuit: sLNv pacemaker network with PDF neuropeptide signaling.
10. NociceptiveAvoidanceCircuit: Thermal/mechanical nociceptive fast escape reflex arc.
"""

from .ring_attractor import EPGRingAttractor
from .motion_detector import HassensteinReichardtEMD
from .escape_circuit import GiantFiberEscapeCircuit
from .learning_circuit import MushroomBodyLearningModule
from .central_pattern_generator import CPGMotorGenerator
from .lif_models import LIFNeuron, IzhikevichNeuron, AdExNeuron
from .antennal_lobe import AntennalLobeCircuit
from .optomotor_response import OptomotorResponseCircuit
from .circadian_clock import CircadianClockCircuit
from .nociceptive_escape import NociceptiveAvoidanceCircuit

__version__ = "1.0.0"

__all__ = [
    "EPGRingAttractor",
    "HassensteinReichardtEMD",
    "GiantFiberEscapeCircuit",
    "MushroomBodyLearningModule",
    "CPGMotorGenerator",
    "LIFNeuron",
    "IzhikevichNeuron",
    "AdExNeuron",
    "AntennalLobeCircuit",
    "OptomotorResponseCircuit",
    "CircadianClockCircuit",
    "NociceptiveAvoidanceCircuit",
]
