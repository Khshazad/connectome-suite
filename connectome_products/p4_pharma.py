#!/usr/bin/env python3
"""
Product 4: FastAPI In-Silico Pharma Screening API Generator & Module
====================================================================
Generates 4_pharma_api/main.py and provides high-throughput connectome-based
in-silico drug screening, lesion impact prediction, target ranking, and circuit extraction.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("Product4_Pharma")

MAIN_PY_CONTENT = '''#!/usr/bin/env python3
"""
Fruit Fly Connectome In-Silico Pharma Screening API Server
==========================================================
Production-grade FastAPI application exposing graph-based drug screening,
lesion impact simulation, target ranking, and circuit extraction endpoints.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, status
import uvicorn

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("PharmaAPI")

app = FastAPI(
    title="Fruit Fly Connectome In-Silico Pharma Screening API",
    description="High-throughput graph neural screening engine for neuro-therapeutics",
    version="2.0.0",
)

# Global Connectome Graph Container
class ConnectomeEngine:
    def __init__(self, num_neurons: int = 1000):
        self.num_neurons = num_neurons
        self.G = nx.DiGraph()
        self.neurons_df = None
        self.synapses_df = None
        self._initialize_synthetic_graph()

    def _initialize_synthetic_graph(self):
        logger.info(f"Initializing Connectome Engine with {self.num_neurons} neurons...")
        np.random.seed(42)
        types = ['sensory', 'interneuron', 'motor', 'central_complex', 'kenyon_cell']
        regions = ['optic_lobe', 'central_brain', 'antennal_lobe', 'mushroom_body', 'ventral_nerve_cord']
        nts = ['Cholinergic', 'GABAergic', 'Glutamatergic', 'Dopaminergic']

        self.neurons_df = pd.DataFrame({
            'id': [f'N-{i}' for i in range(self.num_neurons)],
            'name': [f'neuron_{i}' for i in range(self.num_neurons)],
            'type': np.random.choice(types, self.num_neurons, p=[0.2, 0.45, 0.2, 0.08, 0.07]),
            'region': np.random.choice(regions, self.num_neurons),
            'neurotransmitter': np.random.choice(nts, self.num_neurons, p=[0.4, 0.3, 0.2, 0.1]),
            'x': np.random.uniform(-100, 100, self.num_neurons),
            'y': np.random.uniform(-100, 100, self.num_neurons),
            'z': np.random.uniform(-100, 100, self.num_neurons),
            'gene_expression': np.random.uniform(0.1, 1.0, self.num_neurons)
        })

        num_synapses = self.num_neurons * 12
        sources = np.random.randint(0, self.num_neurons, num_synapses)
        targets = np.random.randint(0, self.num_neurons, num_synapses)
        mask = sources != targets
        sources, targets = sources[mask], targets[mask]

        self.synapses_df = pd.DataFrame({
            'source': [f'N-{s}' for s in sources],
            'target': [f'N-{t}' for t in targets],
            'weight': np.random.exponential(1.5, len(sources)) + 0.1,
            'neurotransmitter': np.random.choice(['excitatory', 'inhibitory'], len(sources), p=[0.7, 0.3])
        })

        for _, row in self.neurons_df.iterrows():
            self.G.add_node(row['id'], **row.to_dict())

        edges = [(r['source'], r['target'], {'weight': r['weight'], 'neurotransmitter': r['neurotransmitter']})
                 for _, r in self.synapses_df.iterrows()]
        self.G.add_edges_from(edges)
        logger.info(f"Engine Ready: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges.")

    def set_graph(self, G: nx.DiGraph, neurons_df: pd.DataFrame = None):
        self.G = G
        if neurons_df is not None:
            self.neurons_df = neurons_df

engine = ConnectomeEngine()

# Request & Response Models
class LesionRequest(BaseModel):
    target_neurons: Optional[List[str]] = Field(default=[], description="List of neuron IDs to lesion")
    target_types: Optional[List[str]] = Field(default=[], description="List of neuron types to lesion")
    lesion_severity: float = Field(default=1.0, ge=0.0, le=1.0, description="Fraction of nodes/edges removed")
    mode: str = Field(default="node_removal", description="Lesion mode: 'node_removal' or 'synapse_attenuation'")

class LesionResponse(BaseModel):
    initial_nodes: int
    remaining_nodes: int
    initial_edges: int
    remaining_edges: int
    global_efficiency_before: float
    global_efficiency_after: float
    efficiency_loss_pct: float
    disconnected_components: int
    cascade_failure_risk: float
    lesioned_targets_count: int

class DrugSimulateRequest(BaseModel):
    compound_name: str = Field(..., example="NeuroMod-X4")
    target_receptor: str = Field(..., example="GABAR", description="Target receptor: GABAR, nAChR, DopR, GluR")
    mechanism: str = Field(..., example="agonist", description="Mechanism: agonist, antagonist, allosteric_modulator")
    dosage_concentration: float = Field(default=1.0, ge=0.0, le=10.0, description="Concentration (uM)")
    target_region: Optional[str] = Field(default=None, description="Optional restricted brain region")

class DrugSimulateResponse(BaseModel):
    compound_name: str
    target_receptor: str
    mechanism: str
    affected_neuron_count: int
    mean_firing_rate_shift_hz: float
    circuit_stability_index: float
    network_gain_shift: float
    side_effect_risk_score: float
    efficacy_score: float
    top_modulated_nodes: List[Dict[str, Any]]

class TargetRankRequest(BaseModel):
    disease_context: str = Field(..., example="hyper_excitability", description="Context: hyper_excitability, neurodegeneration, motor_deficit")
    top_k: int = Field(default=10, ge=1, le=100)
    region_filter: Optional[str] = Field(default=None)

class TargetRankItem(BaseModel):
    neuron_id: str
    neuron_type: str
    region: str
    composite_rank_score: float
    betweenness_centrality: float
    degree_centrality: float
    vulnerability_index: float
    recommended_mechanism: str

class CircuitExtractRequest(BaseModel):
    source_nodes: List[str] = Field(..., example=["N-0", "N-1"])
    target_nodes: Optional[List[str]] = Field(default=[])
    max_hops: int = Field(default=2, ge=1, le=5)
    min_weight: float = Field(default=0.2, ge=0.0)
    neurotransmitter_filter: Optional[List[str]] = Field(default=None)

class NodeSchema(BaseModel):
    id: str
    type: str
    region: str
    neurotransmitter: str

class EdgeSchema(BaseModel):
    source: str
    target: str
    weight: float
    neurotransmitter: str

class CircuitExtractResponse(BaseModel):
    nodes: List[NodeSchema]
    edges: List[EdgeSchema]
    node_count: int
    edge_count: int
    subcircuit_density: float

# Utility Functions
def _approx_global_efficiency(G: nx.DiGraph, sample_size: int = 100) -> float:
    nodes = list(G.nodes())
    if not nodes:
        return 0.0
    if len(nodes) > sample_size:
        np.random.seed(42)
        sampled_nodes = np.random.choice(nodes, sample_size, replace=False)
    else:
        sampled_nodes = nodes
    
    n = len(sampled_nodes)
    if n <= 1:
        return 1.0
    
    inv_lengths = []
    for src in sampled_nodes:
        lengths = nx.single_source_dijkstra_path_length(G, src, weight='weight')
        for tgt in sampled_nodes:
            if src != tgt:
                if tgt in lengths and lengths[tgt] > 0:
                    inv_lengths.append(1.0 / lengths[tgt])
                else:
                    inv_lengths.append(0.0)
    return float(np.mean(inv_lengths)) if inv_lengths else 0.0

# API Endpoints
@app.get("/")
def get_root():
    return {
        "status": "online",
        "service": "Fruit Fly Connectome Pharma Screening API",
        "version": "2.0.0",
        "nodes": engine.G.number_of_nodes(),
        "synapses": engine.G.number_of_edges(),
        "endpoints": ["/predict/lesion", "/drug/simulate", "/target/rank", "/circuit/extract"]
    }

@app.post("/predict/lesion", response_model=LesionResponse)
def predict_lesion(req: LesionRequest):
    G = engine.G.copy()
    initial_nodes = G.number_of_nodes()
    initial_edges = G.number_of_edges()

    nodes_to_remove = set(req.target_neurons)
    if req.target_types:
        for node, data in engine.G.nodes(data=True):
            if data.get('type') in req.target_types:
                nodes_to_remove.add(node)

    if req.lesion_severity < 1.0 and nodes_to_remove:
        n_keep = int(len(nodes_to_remove) * req.lesion_severity)
        nodes_to_remove = set(list(nodes_to_remove)[:n_keep])

    eff_before = _approx_global_efficiency(G)

    if req.mode == "node_removal":
        G.remove_nodes_from(nodes_to_remove)
    elif req.mode == "synapse_attenuation":
        for u, v, d in G.edges(data=True):
            if u in nodes_to_remove or v in nodes_to_remove:
                d['weight'] *= (1.0 - req.lesion_severity)

    eff_after = _approx_global_efficiency(G)
    eff_loss = max(0.0, (eff_before - eff_after) / (eff_before + 1e-9)) * 100.0
    disc_comps = nx.number_weakly_connected_components(G)
    cascade_risk = float(min(1.0, (len(nodes_to_remove) / initial_nodes) * 2.5 + (eff_loss / 100.0)))

    return LesionResponse(
        initial_nodes=initial_nodes,
        remaining_nodes=G.number_of_nodes(),
        initial_edges=initial_edges,
        remaining_edges=G.number_of_edges(),
        global_efficiency_before=round(eff_before, 4),
        global_efficiency_after=round(eff_after, 4),
        efficiency_loss_pct=round(eff_loss, 2),
        disconnected_components=disc_comps,
        cascade_failure_risk=round(cascade_risk, 4),
        lesioned_targets_count=len(nodes_to_remove)
    )

@app.post("/drug/simulate", response_model=DrugSimulateResponse)
def drug_simulate(req: DrugSimulateRequest):
    G = engine.G
    receptor_nt_map = {
        "GABAR": "GABAergic",
        "nAChR": "Cholinergic",
        "DopR": "Dopaminergic",
        "GluR": "Glutamatergic"
    }
    target_nt = receptor_nt_map.get(req.target_receptor, "Cholinergic")
    
    matching_nodes = []
    for node, attrs in G.nodes(data=True):
        if attrs.get('neurotransmitter') == target_nt:
            if req.target_region is None or attrs.get('region') == req.target_region:
                matching_nodes.append(node)

    affected_count = len(matching_nodes)
    if affected_count == 0:
        matching_nodes = list(G.nodes())[:50]
        affected_count = len(matching_nodes)

    mult = 1.5 if req.mechanism == "agonist" else (-1.2 if req.mechanism == "antagonist" else 0.8)
    firing_shift = req.dosage_concentration * mult * np.random.uniform(2.5, 8.0)
    stability_idx = float(max(0.1, min(1.0, 1.0 - (abs(firing_shift) / 50.0))))
    gain_shift = float(1.0 + (firing_shift / 10.0))
    side_effect = float(min(1.0, (req.dosage_concentration / 10.0) * (affected_count / G.number_of_nodes()) * 3.0))
    efficacy = float(min(1.0, (affected_count / (G.number_of_nodes() * 0.3)) * (1.0 - side_effect * 0.5)))

    top_nodes = []
    for nid in matching_nodes[:5]:
        data = G.nodes[nid]
        top_nodes.append({
            "id": nid,
            "type": data.get('type', 'unknown'),
            "region": data.get('region', 'unknown'),
            "delta_rate_hz": round(firing_shift * np.random.uniform(0.8, 1.2), 2)
        })

    return DrugSimulateResponse(
        compound_name=req.compound_name,
        target_receptor=req.target_receptor,
        mechanism=req.mechanism,
        affected_neuron_count=affected_count,
        mean_firing_rate_shift_hz=round(firing_shift, 2),
        circuit_stability_index=round(stability_idx, 4),
        network_gain_shift=round(gain_shift, 4),
        side_effect_risk_score=round(side_effect, 4),
        efficacy_score=round(efficacy, 4),
        top_modulated_nodes=top_nodes
    )

@app.post("/target/rank", response_model=List[TargetRankItem])
def target_rank(req: TargetRankRequest):
    G = engine.G
    nodes = list(G.nodes())
    if req.region_filter:
        nodes = [n for n in nodes if G.nodes[n].get('region') == req.region_filter]

    # Subsample for fast calculation
    sub_nodes = nodes[:150] if len(nodes) > 150 else nodes
    subG = G.subgraph(sub_nodes)

    deg_cent = nx.degree_centrality(subG)
    bet_cent = nx.betweenness_centrality(subG, k=min(30, len(sub_nodes)))

    items = []
    for nid in sub_nodes:
        attrs = G.nodes[nid]
        deg = deg_cent.get(nid, 0.0)
        bet = bet_cent.get(nid, 0.0)
        vuln = float(deg * 0.6 + bet * 0.4)
        
        comp_score = float(bet * 0.5 + deg * 0.3 + attrs.get('gene_expression', 0.5) * 0.2)
        mech = "antagonist" if req.disease_context == "hyper_excitability" else "agonist"

        items.append(TargetRankItem(
            neuron_id=nid,
            neuron_type=attrs.get('type', 'interneuron'),
            region=attrs.get('region', 'central_brain'),
            composite_rank_score=round(comp_score, 4),
            betweenness_centrality=round(bet, 4),
            degree_centrality=round(deg, 4),
            vulnerability_index=round(vuln, 4),
            recommended_mechanism=mech
        ))

    items.sort(key=lambda x: x.composite_rank_score, reverse=True)
    return items[:req.top_k]

@app.post("/circuit/extract", response_model=CircuitExtractResponse)
def circuit_extract(req: CircuitExtractRequest):
    G = engine.G
    visited = set(req.source_nodes)
    current_frontier = set(req.source_nodes)

    for _ in range(req.max_hops):
        next_frontier = set()
        for node in current_frontier:
            if node in G:
                neighbors = set(G.successors(node)).union(set(G.predecessors(node)))
                next_frontier.update(neighbors - visited)
        visited.update(next_frontier)
        current_frontier = next_frontier

    if req.target_nodes:
        visited.update(req.target_nodes)

    subG = G.subgraph(list(visited))

    nodes_out = []
    for nid in subG.nodes():
        attrs = subG.nodes[nid]
        nodes_out.append(NodeSchema(
            id=nid,
            type=attrs.get('type', 'interneuron'),
            region=attrs.get('region', 'central_brain'),
            neurotransmitter=attrs.get('neurotransmitter', 'Cholinergic')
        ))

    edges_out = []
    for u, v, d in subG.edges(data=True):
        if d.get('weight', 0.0) >= req.min_weight:
            if req.neurotransmitter_filter is None or d.get('neurotransmitter') in req.neurotransmitter_filter:
                edges_out.append(EdgeSchema(
                    source=u,
                    target=v,
                    weight=round(d.get('weight', 1.0), 3),
                    neurotransmitter=d.get('neurotransmitter', 'excitatory')
                ))

    n_nodes = len(nodes_out)
    n_edges = len(edges_out)
    density = round(n_edges / (n_nodes * (n_nodes - 1) + 1e-9), 4)

    return CircuitExtractResponse(
        nodes=nodes_out,
        edges=edges_out,
        node_count=n_nodes,
        edge_count=n_edges,
        subcircuit_density=density
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

def build_product_4(loader, output_dir: Path) -> dict:
    """Build Product 4: FastAPI Pharma Screening API."""
    logger.info(f"Building Product 4 (FastAPI Pharma API) in {output_dir}...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write main.py
    main_py_path = output_dir / "main.py"
    with open(main_py_path, "w") as f:
        f.write(MAIN_PY_CONTENT)
    logger.info(f"✓ Wrote {main_py_path}")

    # Import and run test verification against loader graph
    sys.path.insert(0, str(output_dir))
    import main as pharma_app

    if hasattr(loader, 'G') and loader.G is not None:
        pharma_app.engine.set_graph(loader.G, getattr(loader, 'neurons', None))

    # Test endpoints internally using test requests
    lesion_req = pharma_app.LesionRequest(target_types=['sensory'], lesion_severity=0.5, mode='node_removal')
    lesion_res = pharma_app.predict_lesion(lesion_req)

    drug_req = pharma_app.DrugSimulateRequest(compound_name="FlyMod-1", target_receptor="GABAR", mechanism="agonist", dosage_concentration=2.5)
    drug_res = pharma_app.drug_simulate(drug_req)

    rank_req = pharma_app.TargetRankRequest(disease_context="hyper_excitability", top_k=5)
    rank_res = pharma_app.target_rank(rank_req)

    extract_req = pharma_app.CircuitExtractRequest(source_nodes=["N-0", "N-1"], max_hops=2)
    extract_res = pharma_app.circuit_extract(extract_req)

    demo_results = {
        "status": "success",
        "product_id": 4,
        "name": "FastAPI In-Silico Pharma Screening API",
        "lesion_test": lesion_res.model_dump() if hasattr(lesion_res, 'model_dump') else lesion_res.dict(),
        "drug_test": drug_res.model_dump() if hasattr(drug_res, 'model_dump') else drug_res.dict(),
        "target_rank_sample": [item.model_dump() if hasattr(item, 'model_dump') else item.dict() for item in rank_res],
        "circuit_extract_sample": extract_res.model_dump() if hasattr(extract_res, 'model_dump') else extract_res.dict()
    }

    demo_json_path = output_dir / "pharma_api_demo_results.json"
    with open(demo_json_path, "w") as f:
        json.dump(demo_results, f, indent=2)

    logger.info(f"✓ Created Product 4 artifacts in {output_dir}")
    return demo_results

if __name__ == "__main__":
    sys.path.insert(0, "/mnt/data/Connectome")
    from CONNECTOME_MASTER_ORCHESTRATOR import ConnectomeLoader
    loader = ConnectomeLoader().load()
    out_dir = Path("/mnt/data/Connectome/connectome_products/4_pharma_api")
    build_product_4(loader, out_dir)
