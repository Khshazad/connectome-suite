#!/usr/bin/env python3
"""
Product 5: Neuromorphic Hardware & SNN Benchmark Suite Generator & Module
========================================================================
Generates 5_neuromorphic_benchmark/benchmark.py and provides benchmarking for
latency, energy per spike (pJ/spike), dynamic range, throughput (SOPS, GSOPS/W),
biological vs synthetic topology comparisons, and PyNN, Lava, & Nengo schema exports.
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
Evaluates Spiking Neural Network (SNN) performance metrics on connectome & synthetic subgraphs:
- Latency (Time-to-first-spike, Mean ISI, Propagation Delay)
- Energy per Spike (pJ/spike across Intel Loihi 2, BrainChip Akida, SynSense DYNAP-SE, SpiNNaker 2)
- Dynamic Range (Firing rate dynamic range, Weight quantization distortion)
- Throughput (Spikes/sec, SOPS - Synaptic Operations Per Second, GSOPS/Watt)
- Topology Comparison (Biological Connectome vs Erdős-Rényi, Barabási-Albert, Watts-Strogatz)
- Schema Exporters for PyNN 0.10+, Intel Lava, and Nengo SNN
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

HARDWARE_PROFILES = {
    "Intel_Loihi_2": {
        "energy_per_spike_pj": 23.0,
        "static_power_mw": 15.0,
        "max_weight_bits": 8,
        "clock_freq_mhz": 128.0,
        "max_neurons_per_core": 1024,
        "architecture": "Digital Asynchronous Mesh"
    },
    "BrainChip_Akida": {
        "energy_per_spike_pj": 10.0,
        "static_power_mw": 12.0,
        "max_weight_bits": 4,
        "clock_freq_mhz": 100.0,
        "max_neurons_per_core": 2048,
        "architecture": "Neuromorphic Event-Driven Core"
    },
    "SynSense_DYNAP_SE": {
        "energy_per_spike_pj": 5.0,
        "static_power_mw": 5.0,
        "max_weight_bits": 8,
        "clock_freq_mhz": 50.0,
        "max_neurons_per_core": 256,
        "architecture": "Subthreshold Analog Mixed-Signal"
    },
    "SpiNNaker_2": {
        "energy_per_spike_pj": 45.0,
        "static_power_mw": 40.0,
        "max_weight_bits": 16,
        "clock_freq_mhz": 500.0,
        "max_neurons_per_core": 1536,
        "architecture": "ARM Multi-Core Parallel RISC"
    }
}

class LeakyIntegrateAndFireSimulator:
    """Fast vector LIF simulator for benchmark latency & spike extraction."""

    def __init__(self, G: nx.DiGraph, dt_ms: float = 0.5, v_thresh: float = -50.0, v_reset: float = -70.0, tau_m: float = 20.0):
        self.G = G
        self.nodes = list(G.nodes())
        self.N = len(self.nodes)
        self.node2idx = {node: i for i, node in enumerate(self.nodes)}
        self.dt_ms = dt_ms
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.tau_m = tau_m

        self.W = np.zeros((self.N, self.N), dtype=np.float32)
        for u, v, d in G.edges(data=True):
            i, j = self.node2idx[u], self.node2idx[v]
            w = float(d.get('weight', 1.0))
            if d.get('neurotransmitter') == 'inhibitory':
                w = -w
            self.W[i, j] = w

    def run_simulation(self, duration_ms: float = 50.0, input_rate_hz: float = 50.0) -> Dict[str, Any]:
        n_steps = int(duration_ms / self.dt_ms)
        v = np.full(self.N, self.v_reset, dtype=np.float32)
        spike_record = []

        decay = np.exp(-self.dt_ms / self.tau_m)
        poisson_p = (input_rate_hz * (self.dt_ms / 1000.0))

        sensory_indices = [self.node2idx[n] for n in self.nodes if self.G.nodes[n].get('type') == 'sensory']
        if not sensory_indices:
            sensory_indices = list(range(min(20, self.N)))

        for step in range(n_steps):
            ext_stim = (np.random.rand(len(sensory_indices)) < poisson_p) * 15.0
            v[sensory_indices] += ext_stim

            spikes = np.where(v >= self.v_thresh)[0]
            if len(spikes) > 0:
                t_ms = step * self.dt_ms
                for idx in spikes:
                    spike_record.append((t_ms, idx))
                v[spikes] = self.v_reset

                if len(spikes) == 1:
                    synaptic_current = self.W[spikes[0], :]
                else:
                    synaptic_current = np.sum(self.W[spikes, :], axis=0)
                v += synaptic_current * 0.5

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
        nodes = list(G.nodes())
        if len(nodes) > 200:
            sub_nodes = nodes[:200]
            self.G = G.subgraph(sub_nodes).copy()
        else:
            self.G = G

    def benchmark_graph(self, target_G: nx.DiGraph, topology_name: str = "biological", duration_ms: float = 50.0) -> Dict[str, Any]:
        sim = LeakyIntegrateAndFireSimulator(target_G, dt_ms=0.5)
        sim_res = sim.run_simulation(duration_ms=duration_ms)

        spikes = sim_res["spike_record"]
        total_spikes = sim_res["total_spikes"]

        if spikes:
            time_to_first_spike_ms = float(spikes[0][0])
            spike_times = [s[0] for s in spikes]
            isis = np.diff(spike_times)
            mean_isi_ms = float(np.mean(isis)) if len(isis) > 0 else float(duration_ms)
        else:
            time_to_first_spike_ms = float(duration_ms)
            mean_isi_ms = float(duration_ms)

        propagation_latency_ms = time_to_first_spike_ms * 2.5

        weights = [d.get('weight', 1.0) for _, _, d in target_G.edges(data=True)]
        if weights:
            w_max, w_min = max(weights), min(weights)
            dynamic_range_db = float(20 * math.log10((w_max + 1e-6) / (w_min + 1e-6)))
        else:
            dynamic_range_db = 0.0

        sps = float(total_spikes / (duration_ms / 1000.0))
        synapses = target_G.number_of_edges()
        sops = float(total_spikes * synapses / (duration_ms / 1000.0))
        gsops = sops / 1e9

        hw_results = {}
        for hw_name, prof in HARDWARE_PROFILES.items():
            energy_pj = total_spikes * prof["energy_per_spike_pj"]
            total_power_mw = prof["static_power_mw"] + (energy_pj / 1e9) / (duration_ms / 1000.0) * 1000.0
            gsops_per_watt = gsops / (total_power_mw / 1000.0 + 1e-6)

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
            "topology": topology_name,
            "graph_summary": {
                "nodes": target_G.number_of_nodes(),
                "synapses": target_G.number_of_edges(),
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

    def benchmark(self, duration_ms: float = 50.0) -> Dict[str, Any]:
        N = self.G.number_of_nodes()
        E = self.G.number_of_edges()
        p = min(1.0, E / (N * (N - 1) + 1e-9))
        k = max(2, int(E / N))

        bio_res = self.benchmark_graph(self.G, topology_name="biological_connectome", duration_ms=duration_ms)

        G_er = nx.erdos_renyi_graph(n=N, p=p, seed=42, directed=True)
        for u, v in G_er.edges():
            G_er[u][v]['weight'] = np.random.exponential(1.5) + 0.1
        er_res = self.benchmark_graph(G_er, topology_name="synthetic_erdos_renyi", duration_ms=duration_ms)

        G_ba_undir = nx.barabasi_albert_graph(n=N, m=max(1, k // 2), seed=42)
        G_ba = nx.DiGraph(G_ba_undir)
        for u, v in G_ba.edges():
            G_ba[u][v]['weight'] = np.random.exponential(1.5) + 0.1
        ba_res = self.benchmark_graph(G_ba, topology_name="synthetic_barabasi_albert", duration_ms=duration_ms)

        G_ws_undir = nx.watts_strogatz_graph(n=N, k=k if k % 2 == 0 else k + 1, p=0.1, seed=42)
        G_ws = nx.DiGraph(G_ws_undir)
        for u, v in G_ws.edges():
            G_ws[u][v]['weight'] = np.random.exponential(1.5) + 0.1
        ws_res = self.benchmark_graph(G_ws, topology_name="synthetic_watts_strogatz", duration_ms=duration_ms)

        return {
            "graph_summary": bio_res["graph_summary"],
            "latency_metrics": bio_res["latency_metrics"],
            "dynamic_range_metrics": bio_res["dynamic_range_metrics"],
            "throughput_metrics": bio_res["throughput_metrics"],
            "hardware_profiles_evaluation": bio_res["hardware_profiles_evaluation"],
            "topology_comparison": {
                "biological_connectome": bio_res,
                "synthetic_erdos_renyi": er_res,
                "synthetic_barabasi_albert": ba_res,
                "synthetic_watts_strogatz": ws_res
            }
        }

    def export_pynn_schema(self, output_path: Path) -> Dict[str, Any]:
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
        for u, v, d in edges_data[:500]:
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
                    "du": 409,
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

    def export_nengo_schema(self, output_path: Path) -> Dict[str, Any]:
        nodes_data = list(self.G.nodes(data=True))
        edges_data = list(self.G.edges(data=True))

        pop_groups = {}
        for nid, d in nodes_data[:200]:
            ntype = d.get('type', 'interneuron')
            pop_groups.setdefault(ntype, []).append(nid)

        ensembles = []
        for ntype, n_ids in pop_groups.items():
            ensembles.append({
                "name": f"Ens_{ntype}",
                "n_neurons": len(n_ids),
                "dimensions": 1,
                "neuron_type": "LIFRate",
                "tau_rc": 0.02,
                "tau_ref": 0.002,
                "neuron_ids": n_ids
            })

        connections = []
        for u, v, d in edges_data[:300]:
            connections.append({
                "pre": f"Ens_{self.G.nodes[u].get('type', 'interneuron')}",
                "post": f"Ens_{self.G.nodes[v].get('type', 'interneuron')}",
                "transform": round(float(d.get('weight', 1.0)) * 0.1, 4),
                "synapse": 0.005
            })

        nengo_data = {
            "nengo_version": "3.2.0",
            "target_backend": "nengo_dl / nengo_loihi",
            "description": "Fruit Fly Connectome Nengo SNN Ensemble Specification",
            "ensembles": ensembles,
            "connections": connections
        }

        with open(output_path, "w") as f:
            json.dump(nengo_data, f, indent=2)
        logger.info(f"✓ Exported Nengo schema to {output_path}")
        return nengo_data

if __name__ == "__main__":
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
    benchmark_res = suite.benchmark(duration_ms=50.0)

    pynn_path = output_dir / "pynn_schema.json"
    pynn_schema = suite.export_pynn_schema(pynn_path)

    lava_path = output_dir / "lava_schema.json"
    lava_schema = suite.export_lava_schema(lava_path)

    nengo_path = output_dir / "nengo_schema.json"
    nengo_schema = suite.export_nengo_schema(nengo_path)

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
        "nengo_schema_path": str(nengo_path),
        "report_path": str(report_path),
        "metrics_summary": benchmark_res["throughput_metrics"]
    }

if __name__ == "__main__":
    sys.path.insert(0, "/mnt/data/Connectome")
    from CONNECTOME_MASTER_ORCHESTRATOR import ConnectomeLoader
    loader = ConnectomeLoader().load()
    out_dir = Path("/mnt/data/Connectome/connectome_products/5_neuromorphic_benchmark")
    build_product_5(loader, out_dir)
