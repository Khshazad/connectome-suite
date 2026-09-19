#!/usr/bin/env python3
"""
Product 5: Neuromorphic Hardware & SNN Benchmark Suite Generator & Module
========================================================================
Generates 5_neuromorphic_benchmark/benchmark.py and provides benchmarking for
latency, energy per spike (pJ/spike), dynamic range, throughput, and PyNN & Lava schema exports.
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Product5_Benchmark")

BENCHMARK_PY_CONTENT = '''#!/usr/bin/env python3
"""
Neuromorphic SNN Hardware Benchmark Engine & Framework
======================================================
Evaluates Spiking Neural Network (SNN) performance metrics on connectome subgraphs:
- Latency (Time-to-first-spike, Mean ISI, Propagation Delay)
- Energy per Spike (pJ/spike across Loihi 2, SpiNNaker 2, TrueNorth, BrainScaleS-2)
- Dynamic Range (Firing rate dynamic range, Weight quantization distortion)
- Throughput (Spikes/sec, SOPS - Synaptic Operations Per Second, GSOPS/Watt)
- Schema Exporters for PyNN 0.10+ and Intel Lava
"""

import os
import sys
import json
import logging
import math
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("NeuromorphicBenchmark")

# Neuromorphic Hardware Profiles
HARDWARE_PROFILES = {
    "Loihi_2": {
        "energy_per_spike_pj": 23.0,
        "static_power_mw": 15.0,
        "max_weight_bits": 8,
        "clock_freq_mhz": 128.0,
        "max_neurons_per_core": 1024,
        "architecture": "Digital Asynchronous Mesh"
    },
    "SpiNNaker_2": {
        "energy_per_spike_pj": 45.0,
        "static_power_mw": 40.0,
        "max_weight_bits": 16,
        "clock_freq_mhz": 500.0,
        "max_neurons_per_core": 1536,
        "architecture": "ARM Multi-Core Parallel RISC"
    },
    "TrueNorth": {
        "energy_per_spike_pj": 26.0,
        "static_power_mw": 70.0,
        "max_weight_bits": 4,
        "clock_freq_mhz": 1.0,
        "max_neurons_per_core": 256,
        "architecture": "Crossbar Digital Array"
    },
    "BrainScaleS_2": {
        "energy_per_spike_pj": 120.0,
        "static_power_mw": 120.0,
        "max_weight_bits": 6,
        "clock_freq_mhz": 10.0, # Accelerated 10^4 x physical time
        "max_neurons_per_core": 512,
        "architecture": "Analog Mixed-Signal Subthreshold"
    }
}

class LeakyIntegrateAndFireSimulator:
    """Fast vector LIF simulator for benchmark latency & spike extraction."""

    def __init__(self, G: nx.DiGraph, dt_ms: float = 0.1, v_thresh: float = -50.0, v_reset: float = -70.0, tau_m: float = 20.0):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.node2idx = {node: i for i, node in enumerate(self.nodes)}
        self.dt_ms = dt_ms
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_m = tau_m

        # Build adjacency weight matrix
        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        for u, v, d in G.edges(data=True):
            i, j = self.node2idx[u], self.node2idx[v]
            w = float(d.get('weight', 1.0))
            if d.get('neurotransmitter') == 'inhibitory':
                w = -w
            self.W[i, j] = w

    def run_simulation(self, duration_ms: float = 100.0, input_rate_hz: float = 50.0) -> Dict[str, Any]:
        n_steps = int(duration_ms / self.dt_ms)
        v = np.full(self.N, self.v_reset, dtype=np.float32)
        spike_record = [] # (step, neuron_idx)

        decay = np.exp(-self.dt_ms / self.tau_m)
        poisson_p = (input_rate_hz * (self.dt_ms / 1000.0))

        # Sensory neurons get external stimulation
        sensory_indices = [self.node2idx[n] for n in self.nodes if self.G.nodes[n].get('type') == 'sensory']
        if not sensory_indices:
            sensory_indices = list(range(min(20, self.N)))

        for step in range(n_steps):
            # Poisson input
            ext_stim = (np.random.rand(len(sensory_indices)) < poisson_p) * 15.0
            v[sensory_indices] += ext_stim

            # Check threshold
            spikes = np.where(v >= self.v_thresh)[0]
            if len(spikes) > 0:
                for idx in spikes:
                    spike_record.append((step * self.dt_ms, idx))
                v[spikes] = self.v_reset

                # Recurrent synaptic propagation
                if len(spikes) == 1:
                    synaptic_current = self.W[spikes[0], :]
                else:
                    synaptic_current = np.sum(self.W[spikes, :], axis=0)
                v += synaptic_current * 0.5

            # Membrane voltage decay
            v = self.v_reset + (v - self.v_reset) * decay

        return {
            "spike_record": spike_record,
            "total_spikes": len(spike_record),
            "n_steps": n_steps,
            "duration_ms": duration_ms
        }

class NeuromorphicBenchmarkSuite:
    """Core Neuromorphic Hardware Benchmark Evaluator."""

    def __init__(self, G: nx.DiGraph):
        self.G = G

    def benchmark(self, duration_ms: float = 100.0) -> Dict[str, Any]:
        sim = LeakyIntegrateAndFireSimulator(self.G)
        sim_res = sim.run_simulation(duration_ms=duration_ms)

        spikes = sim_res["spike_record"]
        total_spikes = sim_res["total_spikes"]

        # Latency Metrics
        if spikes:
            time_to_first_spike_ms = float(spikes[0][0])
            spike_times = [s[0] for s in spikes]
            isis = np.diff(spike_times)
            mean_isi_ms = float(np.mean(isis)) if len(isis) > 0 else float(duration_ms)
        else:
            time_to_first_spike_ms = float(duration_ms)
            mean_isi_ms = float(duration_ms)

        # Graph propagation depth approximation
        avg_path_len = float(nx.average_shortest_path_length(self.G.to_undirected())) if nx.is_connected(self.G.to_undirected()) else 3.5
        propagation_latency_ms = time_to_first_spike_ms * avg_path_len

        # Dynamic Range & Quantization Metrics
        weights = [d.get('weight', 1.0) for _, _, d in self.G.edges(data=True)]
        if weights:
            w_max, w_min = max(weights), min(weights)
            dynamic_range_db = float(20 * math.log10((w_max + 1e-6) / (w_min + 1e-6)))
        else:
            dynamic_range_db = 0.0

        # Throughput Metrics
        sps = float(total_spikes / (duration_ms / 1000.0))
        synapses = self.G.number_of_edges()
        sops = float(total_spikes * synapses / (duration_ms / 1000.0))
        gsops = sops / 1e9

        # Hardware Target Evaluation
        hw_results = {}
        for hw_name, prof in HARDWARE_PROFILES.items():
            energy_pj = total_spikes * prof["energy_per_spike_pj"]
            static_energy_mj = (prof["static_power_mw"] * (duration_ms / 1000.0))
            total_power_mw = prof["static_power_mw"] + (energy_pj / 1e9) / (duration_ms / 1000.0) * 1000.0
            gsops_per_watt = gsops / (total_power_mw / 1000.0 + 1e-6)

            # Weight quantization loss (MSE)
            bits = prof["max_weight_bits"]
            levels = 2 ** bits
            q_weights = np.round(np.array(weights) / (max(weights) + 1e-6) * (levels - 1)) / (levels - 1) * max(weights) if weights else np.array([])
            q_error_mse = float(np.mean((np.array(weights) - q_weights) ** 2)) if len(weights) > 0 else 0.0

            hw_results[hw_name] = {
                "energy_per_spike_pj": prof["energy_per_spike_pj"],
                "total_dynamic_energy_nj": round(energy_pj / 1000.0, 3),
                "total_power_mw": round(total_power_mw, 3),
                "gsops_per_watt": round(gsops_per_watt, 3),
                "quantization_bits": bits,
                "weight_quantization_mse": round(q_error_mse, 6)
            }

        return {
            "graph_summary": {
                "nodes": self.G.number_of_nodes(),
                "synapses": self.G.number_of_edges(),
            },
            "latency_metrics": {
                "time_to_first_spike_ms": round(time_to_first_spike_ms, 3),
                "mean_isi_ms": round(mean_isi_ms, 3),
                "estimated_propagation_latency_ms": round(propagation_latency_ms, 3)
            },
            "dynamic_range_metrics": {
                "weight_dynamic_range_db": round(dynamic_range_db, 2),
                "min_weight": round(min(weights), 4) if weights else 0,
                "max_weight": round(max(weights), 4) if weights else 0
            },
            "throughput_metrics": {
                "total_spikes": total_spikes,
                "spikes_per_second": round(sps, 2),
                "synaptic_ops_per_second_SOPS": round(sops, 2),
                "GSOPS": round(gsops, 4)
            },
            "hardware_profiles_evaluation": hw_results
        }

    def export_pynn_schema(self, output_path: Path) -> Dict[str, Any]:
        """Export connectome subgraph to PyNN 0.10+ schema JSON."""
        nodes_data = list(self.G.nodes(data=True))
        edges_data = list(self.G.edges(data=True))

        pop_groups = {}
        for nid, d in nodes_data:
            ntype = d.get('type', 'interneuron')
            pop_groups.setdefault(ntype, []).append(nid)

        populations = []
        for p_name, n_ids in pop_groups.items():
            populations.append({
                "label": p_name,
                "celltype": "IF_curr_exp",
                "size": len(n_ids),
                "parameters": {
                    "v_rest": -70.0,
                    "v_thresh": -50.0,
                    "v_reset": -70.0,
                    "cm": 1.0,
                    "tau_m": 20.0,
                    "tau_syn_E": 5.0,
                    "tau_syn_I": 5.0
                },
                "neuron_ids": n_ids
            })

        projections = []
        for u, v, d in edges_data[:500]: # Export top 500 connections for compactness
            projections.append({
                "presynaptic": u,
                "postsynaptic": v,
                "weight": round(float(d.get('weight', 1.0)), 4),
                "delay": 1.0,
                "synapse_type": "StaticSynapse",
                "receptive_type": "excitatory" if d.get('neurotransmitter') != 'inhibitory' else "inhibitory"
            })

        pynn_data = {
            "pynn_version": "0.10.0",
            "simulator": "nest/brian2/spynnaker",
            "description": "Fruit Fly Connectome PyNN SNN Population Specification",
            "populations": populations,
            "projections": projections
        }

        with open(output_path, "w") as f:
            json.dump(pynn_data, f, indent=2)
        logger.info(f"✓ Exported PyNN schema to {output_path}")
        return pynn_data

    def export_lava_schema(self, output_path: Path) -> Dict[str, Any]:
        """Export connectome subgraph to Intel Lava Neuromorphic Architecture JSON."""
        nodes_data = list(self.G.nodes(data=True))
        edges_data = list(self.G.edges(data=True))

        processes = []
        for nid, d in nodes_data[:200]:
            processes.append({
                "name": f"LIFProcess_{nid}",
                "process_type": "Lava.Proc.LIF",
                "in_ports": ["a_in"],
                "out_ports": ["s_out"],
                "vars": {
                    "v": -70.0,
                    "u": 0.0,
                    "vth": -50.0,
                    "du": 409, # Quantized decay
                    "dv": 204
                }
            })

        connections = []
        for u, v, d in edges_data[:300]:
            connections.append({
                "src_process": f"LIFProcess_{u}",
                "src_port": "s_out",
                "dst_process": f"LIFProcess_{v}",
                "dst_port": "a_in",
                "weight": int(np.clip(float(d.get('weight', 1.0)) * 16, -128, 127))
            })

        lava_data = {
            "lava_version": "0.8.0",
            "target_hardware": "Intel Loihi 2",
            "description": "Fruit Fly Connectome Lava Network Specification",
            "processes": processes,
            "connections": connections
        }

        with open(output_path, "w") as f:
            json.dump(lava_data, f, indent=2)
        logger.info(f"✓ Exported Lava schema to {output_path}")
        return lava_data

if __name__ == "__main__":
    # Internal test execution
    print("Neuromorphic Benchmark Engine Loaded.")
'''

def build_product_5(loader, output_dir: Path) -> dict:
    """Build Product 5: Neuromorphic Hardware Benchmark Suite."""
    logger.info(f"Building Product 5 (Neuromorphic Benchmark) in {output_dir}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write benchmark.py
    bench_py_path = output_dir / "benchmark.py"
    with open(bench_py_path, "w") as f:
        f.write(BENCHMARK_PY_CONTENT)
    logger.info(f"✓ Wrote {bench_py_path}")

    # Import and run benchmark against loader graph
    sys.path.insert(0, str(output_dir))
    import benchmark as bench_module

    suite = bench_module.NeuromorphicBenchmarkSuite(loader.G)
    benchmark_res = suite.benchmark(duration_ms=100.0)

    pynn_path = output_dir / "pynn_schema.json"
    pynn_schema = suite.export_pynn_schema(pynn_path)

    lava_path = output_dir / "lava_schema.json"
    lava_schema = suite.export_lava_schema(lava_path)

    report_path = output_dir / "neuromorphic_benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(benchmark_res, f, indent=2)

    logger.info(f"✓ Created Product 5 artifacts in {output_dir}")
    return {
        "status": "success",
        "product_id": 5,
        "name": "Neuromorphic SNN Benchmark Suite",
        "pynn_schema_path": str(pynn_path),
        "lava_schema_path": str(lava_path),
        "report_path": str(report_path),
        "metrics_summary": benchmark_res["throughput_metrics"]
    }

if __name__ == "__main__":
    sys.path.insert(0, "/mnt/data/Connectome")
    from CONNECTOME_MASTER_ORCHESTRATOR import ConnectomeLoader
    loader = ConnectomeLoader().load()
    out_dir = Path("/mnt/data/Connectome/connectome_products/5_neuromorphic_benchmark")
    build_product_5(loader, out_dir)
