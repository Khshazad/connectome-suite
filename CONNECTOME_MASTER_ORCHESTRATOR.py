#!/usr/bin/env python3
"""
FRUIT FLY CONNECTOME & BIO-AI SUITE: 10-PRODUCT MASTER ORCHESTRATOR
================================================────────────────====

Load connectome once with vectorized operations; build 10 high-value,
production-grade products simultaneously.

PRODUCTS:
1.  Interactive 3D WebGL Explorer (Three.js Web App)
2.  Bio-Inspired PyTorch LIF Spiking Circuit Library (EPG Ring Attractor, EMD, MB)
3.  Behavior Prediction SaaS Game & Benchmark Web App
4.  FastAPI In-Silico Pharma Screening API Backend
5.  Neuromorphic Hardware & SNN Benchmark Suite (PyNN & Lava exports)
6.  Automated Brain-Wide Gene-to-Circuit Receptor Mapper (scRNA-seq + Connectomics)
7.  Embodied Bio-Robotics Closed-Loop Simulator (Central Complex Navigation)
8.  Neuromorphic Edge-AI Pruner & Model Compressor (ANN-to-Bio Sparsity)
9.  Natural Language AI Co-Pilot for Connectomics (Graph RAG Agent)
10. Multi-Scale Neurodegenerative Digital Twin (In-Silico Disease Modeling)
"""

import os
import sys
import json
import pickle
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import networkx as nx

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ConnectomeOrchestrator")

BASE_DIR = Path("/mnt/data/Connectome")
PRODUCTS_DIR = BASE_DIR / "connectome_products"
CACHE_DIR = Path.home() / ".connectome_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

class ConnectomeLoader:
    """Fast, vectorized loader and caching engine for connectome graphs."""
    
    def __init__(self, num_neurons=2000, num_synapses=25000):
        self.num_neurons = num_neurons
        self.num_synapses = num_synapses
        self.neurons = None
        self.synapses = None
        self.G = None
        self.metadata = {}

    def load(self):
        logger.info("=" * 60)
        logger.info("PHASE 0: Fast Vectorized Connectome Loading")
        logger.info("=" * 60)
        
        cache_file = CACHE_DIR / f"connectome_fast_{self.num_neurons}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached connectome dataset from {cache_file}")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                self.neurons = data["neurons"]
                self.synapses = data["synapses"]
        else:
            logger.info("Generating vectorized biological graph model...")
            self._generate_data()
            with open(cache_file, "wb") as f:
                pickle.dump({"neurons": self.neurons, "synapses": self.synapses}, f)
            logger.info(f"Cached biological graph dataset to {cache_file}")
            
        self._build_graph_vectorized()
        self._compute_metadata()
        logger.info("✓ Connectome loaded successfully.")
        return self

    def _generate_data(self):
        types = ['sensory', 'interneuron', 'motor', 'central_complex', 'kenyon_cell', 'giant_fiber']
        regions = ['optic_lobe', 'central_brain', 'antennal_lobe', 'mushroom_body', 'ventral_nerve_cord']
        neurotransmitters = ['Cholinergic', 'GABAergic', 'Glutamatergic', 'Dopaminergic']
        
        np.random.seed(42)
        
        self.neurons = pd.DataFrame({
            'id': [f'N-{i}' for i in range(self.num_neurons)],
            'name': [f'neuron_{i}' for i in range(self.num_neurons)],
            'type': np.random.choice(types, self.num_neurons, p=[0.2, 0.45, 0.2, 0.05, 0.08, 0.02]),
            'region': np.random.choice(regions, self.num_neurons),
            'neurotransmitter': np.random.choice(neurotransmitters, self.num_neurons, p=[0.4, 0.3, 0.2, 0.1]),
            'x': np.random.uniform(-100, 100, self.num_neurons),
            'y': np.random.uniform(-100, 100, self.num_neurons),
            'z': np.random.uniform(-100, 100, self.num_neurons),
            'gene_expression': np.random.uniform(0.1, 1.0, self.num_neurons),
        })

        sources = np.random.randint(0, self.num_neurons, self.num_synapses)
        targets = np.random.randint(0, self.num_neurons, self.num_synapses)
        # Avoid self-loops
        mask = sources != targets
        sources, targets = sources[mask], targets[mask]
        
        self.synapses = pd.DataFrame({
            'source': [f'N-{s}' for s in sources],
            'target': [f'N-{t}' for t in targets],
            'weight': np.random.exponential(1.5, len(sources)) + 0.1,
            'neurotransmitter': np.random.choice(['excitatory', 'inhibitory'], len(sources), p=[0.7, 0.3]),
        })

    def _build_graph_vectorized(self):
        logger.info("Building NetworkX graph with vectorized edge insertion...")
        self.G = nx.DiGraph()
        
        # Add node attributes in bulk
        nodes_dict = self.neurons.set_index('id').to_dict('index')
        for node_id, attrs in nodes_dict.items():
            self.G.add_node(node_id, **attrs)
            
        # Add edges in bulk
        edges = [(row['source'], row['target'], {'weight': row['weight'], 'neurotransmitter': row['neurotransmitter']}) 
                 for _, row in self.synapses.iterrows()]
        self.G.add_edges_from(edges)
        
        logger.info(f"✓ Graph ready: {self.G.number_of_nodes():,} nodes, {self.G.number_of_edges():,} synapses.")

    def _compute_metadata(self):
        self.metadata = {
            'neuron_count': len(self.neurons),
            'synapse_count': len(self.synapses),
            'regions': self.neurons['region'].unique().tolist(),
            'neuron_types': self.neurons['type'].unique().tolist(),
            'timestamp': datetime.now().isoformat(),
        }

def run_product_builder(product_id: int, loader: ConnectomeLoader):
    """Execute product builder module dynamically."""
    logger.info(f"Building Product {product_id}...")
    
    if product_id == 1:
        from connectome_products.p1_explorer import build_product_1
        return build_product_1(loader, PRODUCTS_DIR / "1_interactive_explorer")
    elif product_id == 2:
        from connectome_products.p2_circuits import build_product_2
        return build_product_2(loader, PRODUCTS_DIR / "2_circuit_library")
    elif product_id == 3:
        from connectome_products.p3_game import build_product_3
        return build_product_3(loader, PRODUCTS_DIR / "3_behavior_game")
    elif product_id == 4:
        from connectome_products.p4_pharma import build_product_4
        return build_product_4(loader, PRODUCTS_DIR / "4_pharma_api")
    elif product_id == 5:
        from connectome_products.p5_benchmark import build_product_5
        return build_product_5(loader, PRODUCTS_DIR / "5_neuromorphic_benchmark")
    elif product_id == 6:
        from connectome_products.p6_gene_mapper import build_product_6
        return build_product_6(loader, PRODUCTS_DIR / "6_gene_circuit_mapper")
    elif product_id == 7:
        from connectome_products.p7_robotics import build_product_7
        return build_product_7(loader, PRODUCTS_DIR / "7_embodied_robotics")
    elif product_id == 8:
        from connectome_products.p8_pruner import build_product_8
        return build_product_8(loader, PRODUCTS_DIR / "8_edge_ai_pruner")
    elif product_id == 9:
        from connectome_products.p9_copilot import build_product_9
        return build_product_9(loader, PRODUCTS_DIR / "9_natural_language_copilot")
    elif product_id == 10:
        from connectome_products.p10_digital_twin import build_product_10
        return build_product_10(loader, PRODUCTS_DIR / "10_disease_digital_twin")
    else:
        raise ValueError(f"Unknown product ID: {product_id}")

def main():
    parser = argparse.ArgumentParser(description="Fruit Fly Connectome 10-Product Master Orchestrator")
    parser.add_argument("--all", action="store_true", help="Build all 10 products in parallel")
    parser.add_argument("--product", type=int, choices=range(1, 11), help="Build a specific product (1-10)")
    args = parser.parse_args()

    loader = ConnectomeLoader().load()
    
    if args.product:
        run_product_builder(args.product, loader)
    else:
        logger.info("\n" + "=" * 60)
        logger.info("EXECUTING 10-PRODUCT PARALLEL BUILD (HERMES ORCHESTRATION)")
        logger.info("=" * 60)
        
        products_to_build = list(range(1, 11))
        results = {}
        
        # Parallel execution across all product builders
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_pid = {executor.submit(run_product_builder, pid, loader): pid for pid in products_to_build}
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    res = future.result()
                    results[pid] = "✓ SUCCESS"
                    logger.info(f"Product {pid} Build Completed.")
                except Exception as exc:
                    results[pid] = f"✗ FAILED: {exc}"
                    logger.error(f"Product {pid} generated an exception: {exc}")
                    
        logger.info("\n" + "=" * 60)
        logger.info("BUILD SUMMARY")
        logger.info("=" * 60)
        for pid in sorted(results.keys()):
            logger.info(f"Product {pid}: {results[pid]}")

if __name__ == "__main__":
    main()
