"""
models/__init__.py
==================
Public exports for the NQAL neural core.
"""

from models.encoding import SpikeEncoder, TLSTypeMask
from models.lif_neuron import LIFNeuron, LIFState, SurrogateSpike, STDPLearner
from models.snn import SNN, leaky_integral
from models.dsgat import DSGAT, DSGATLayer
from models.qni import QNI, QuantumRandomWalk
from models.zkp import ZKPLearn, Groth16Simulator            # Part 4
from models.federated import SwarmFederatedLearning           # Part 4
from models.gan import ThreatImmunizationGAN, Generator, Discriminator  # Part 4
from models.nqal_pipeline import NQALPipeline                 # Part 4

__all__ = [
    "SpikeEncoder", "TLSTypeMask",
    "LIFNeuron", "LIFState", "SurrogateSpike", "STDPLearner",
    "SNN", "leaky_integral",
    "DSGAT", "DSGATLayer",
    "QNI", "QuantumRandomWalk",
    "ZKPLearn", "Groth16Simulator",
    "SwarmFederatedLearning",
    "ThreatImmunizationGAN", "Generator", "Discriminator",
    "NQALPipeline",
]
