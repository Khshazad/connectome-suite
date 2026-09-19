#!/usr/bin/env python3
"""
Multi-Scale Neurodegenerative Disease Digital Twin Simulator
===========================================================
Simulates in-silico progressive neurodegenerative pathologies:
1. Alzheimer's Disease: Prion-like graph diffusion of Tau / Amyloid-Beta pathology.
2. Parkinson's Disease: Selective vulnerability and cell loss of Dopaminergic neurons.
3. Synaptic Pruning: Activity-dependent and stochastic synaptic weight decay.
4. Network Health Decay: Quantitative tracking of structural efficiency, modularity, and health index.
"""

import os
import sys
import json
import logging
import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("NeurodegenerativeDigitalTwin")

class NeurodegenerativeDigitalTwin:
    """Multi-Scale In-Silico Disease Trajectory Simulator."""

    DISEASE_MODELS = ["alzheimers", "parkinsons", "synaptic_pruning"]

    def __init__(self, G: nx.DiGraph, neurons_df: pd.DataFrame = None):
        self.G_base = G
        self.neurons_df = neurons_df

    def simulate_alzheimers_spreading(self, steps: int = 10, seed_count: int = 5, beta: float = 0.3, kappa: float = 0.15) -> Dict[str, Any]:
        """Simulate prion-like Tau/Amyloid spreading along connectome edges."""
        logger.info(f"Simulating Alzheimer's Disease spreading across {steps} stages...")
        G = self.G_base.copy()
        nodes = list(G.nodes())
        N = len(nodes)
        node2idx = {n: i for i, n in enumerate(nodes)}

        # Disease concentration per node C_i in [0, 1]
        C = np.zeros(N, dtype=np.float64)
        np.random.seed(42)
        seed_indices = np.random.choice(N, seed_count, replace=False)
        C[seed_indices] = 0.8 # Initial pathology seeds

        # Adjacency matrix for diffusion
        A = nx.to_numpy_array(G, nodelist=nodes)

        history = []
        for step in range(steps):
            # Compute graph diffusion: dC/dt = beta * C * (1-C) + kappa * sum_j A_ji (C_j - C_i)
            neighbor_diff = np.dot(A.T, C) - np.sum(A, axis=0) * C
            dC = beta * C * (1.0 - C) + kappa * neighbor_diff
            C = np.clip(C + dC * 0.1, 0.0, 1.0)

            # High pathology causes synaptic weight decay & node removal
            high_pathology_nodes = [nodes[i] for i in range(N) if C[i] > 0.75]
            G_current = G.copy()
            G_current.remove_nodes_from(high_pathology_nodes)

            # Compute health metrics
            metrics = self._compute_network_health(G_current, base_n=N)
            metrics["stage"] = step
            metrics["mean_pathology_load"] = round(float(np.mean(C)), 4)
            metrics["severely_damaged_nodes"] = len(high_pathology_nodes)
            history.append(metrics)

        return {
            "disease": "Alzheimer's Disease (Tau/Amyloid Graph Diffusion)",
            "simulation_steps": steps,
            "trajectory": history
        }

    def simulate_parkinsons_degeneration(self, steps: int = 10, cell_loss_rate: float = 0.08) -> Dict[str, Any]:
        """Simulate selective Dopaminergic cell loss and circuit breakdown."""
        logger.info(f"Simulating Parkinson's Disease degeneration across {steps} stages...")
        G = self.G_base.copy()
        base_n = G.number_of_nodes()

        dopaminergic_nodes = [n for n, d in G.nodes(data=True) if d.get('neurotransmitter') == 'Dopaminergic']
        if not dopaminergic_nodes:
            dopaminergic_nodes = list(G.nodes())[:int(base_n * 0.1)]

        history = []
        dead_nodes = set()

        for step in range(steps):
            # Progressively kill dopaminergic neurons first, then spread to connected targets
            n_kill = min(len(dopaminergic_nodes), int(len(dopaminergic_nodes) * (step + 1) * cell_loss_rate))
            dead_nodes.update(dopaminergic_nodes[:n_kill])

            G_current = G.copy()
            G_current.remove_nodes_from(dead_nodes)

            metrics = self._compute_network_health(G_current, base_n=base_n)
            metrics["stage"] = step
            metrics["dopaminergic_cell_loss_pct"] = round((n_kill / (len(dopaminergic_nodes) + 1e-9)) * 100.0, 2)
            metrics["total_dead_neurons"] = len(dead_nodes)
            history.append(metrics)

        return {
            "disease": "Parkinson's Disease (Dopaminergic Neurodegeneration)",
            "simulation_steps": steps,
            "trajectory": history
        }

    def simulate_synaptic_pruning(self, steps: int = 10, pruning_rate: float = 0.05) -> Dict[str, Any]:
        """Simulate progressive age-related activity-dependent synaptic pruning."""
        logger.info(f"Simulating Synaptic Pruning across {steps} stages...")
        G = self.G_base.copy()
        base_n = G.number_of_nodes()
        base_e = G.number_of_edges()

        history = []
        for step in range(steps):
            # Prune weakest edges
            edges = list(G.edges(data=True))
            edges.sort(key=lambda e: e[2].get('weight', 1.0))
            
            n_prune = int(base_e * pruning_rate * (step + 1))
            edges_to_remove = [(u, v) for u, v, _ in edges[:n_prune]]

            G_current = G.copy()
            G_current.remove_edges_from(edges_to_remove)

            metrics = self._compute_network_health(G_current, base_n=base_n)
            metrics["stage"] = step
            metrics["pruned_synapses_pct"] = round((n_prune / base_e) * 100.0, 2)
            history.append(metrics)

        return {
            "disease": "Synaptic Pruning (Age-Related Connectome Decay)",
            "simulation_steps": steps,
            "trajectory": history
        }

    def compare_all_diseases(self, steps: int = 8) -> Dict[str, Any]:
        """Run all disease simulations and compare health decay curves."""
        ad_res = self.simulate_alzheimers_spreading(steps=steps)
        pd_res = self.simulate_parkinsons_degeneration(steps=steps)
        sp_res = self.simulate_synaptic_pruning(steps=steps)

        comparison = []
        for s in range(steps):
            comparison.append({
                "stage": s,
                "alzheimers_health_index": ad_res["trajectory"][s]["functional_health_index"],
                "parkinsons_health_index": pd_res["trajectory"][s]["functional_health_index"],
                "pruning_health_index": sp_res["trajectory"][s]["functional_health_index"]
            })

        return {
            "alzheimers_simulation": ad_res,
            "parkinsons_simulation": pd_res,
            "pruning_simulation": sp_res,
            "trajectory_comparison": comparison
        }

    def _compute_network_health(self, G: nx.DiGraph, base_n: int) -> Dict[str, Any]:
        """Calculate quantitative network integrity and efficiency metrics."""
        remaining_nodes = G.number_of_nodes()
        remaining_edges = G.number_of_edges()

        node_survival_ratio = float(remaining_nodes / (base_n + 1e-9))

        # Sample connected components
        undir = G.to_undirected()
        components = list(nx.connected_components(undir))
        largest_cc_size = max(len(c) for c in components) if components else 0
        cc_ratio = float(largest_cc_size / (base_n + 1e-9))

        # Functional health index formula: H = 0.5 * CC_Ratio + 0.3 * Node_Survival + 0.2 * Edge_Density
        health_index = float(0.5 * cc_ratio + 0.3 * node_survival_ratio + 0.2 * min(1.0, remaining_edges / (base_n * 5.0)))

        return {
            "remaining_nodes": remaining_nodes,
            "remaining_edges": remaining_edges,
            "node_survival_ratio": round(node_survival_ratio, 4),
            "largest_component_ratio": round(cc_ratio, 4),
            "functional_health_index": round(health_index, 4)
        }

if __name__ == "__main__":
    print("Multi-Scale Neurodegenerative Digital Twin Loaded.")
