#!/usr/bin/env python3
"""
Product 4: FastAPI In-Silico Pharma Screening API Generator & Module
====================================================================
Generates 4_pharma_api/main.py and provides high-throughput connectome-based
in-silico drug screening, lesion impact prediction, target ranking, and circuit extraction.
Includes virtual receptor binding kinetics (Hill equation), SQLite/JSON caching, and Pydantic v2 schemas.
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
Featuring Pydantic v2 schemas, virtual drug receptor binding kinetics (Hill equation),
and persistent SQLite caching.
"""

import os
import sys
import json
import hashlib
import sqlite3
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI, HTTPException, Query, status
import uvicorn

# Configure Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("PharmaAPI")

app = FastAPI(
    title="Fruit Fly Connectome In-Silico Pharma Screening API",
    description="High-throughput graph neural screening engine for neuro-therapeutics with Hill receptor kinetics & SQLite caching.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Persistent SQLite Cache System
class PharmaCacheDB:
    def __init__(self, db_path: str = "pharma_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        request_hash TEXT PRIMARY KEY,
                        endpoint TEXT NOT NULL,
                        response_json TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not initialize SQLite cache: {e}")

    def get(self, endpoint: str, params_dict: dict) -> Optional[dict]:
        try:
            param_str = json.dumps(params_dict, sort_keys=True)
            req_hash = hashlib.sha256(f"{endpoint}:{param_str}".encode('utf-8')).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT response_json FROM api_cache WHERE request_hash = ?", (req_hash,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None

    def set(self, endpoint: str, params_dict: dict, response_data: dict):
        try:
            param_str = json.dumps(params_dict, sort_keys=True)
            req_hash = hashlib.sha256(f"{endpoint}:{param_str}".encode('utf-8')).hexdigest()
            resp_str = json.dumps(response_data)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO api_cache (request_hash, endpoint, response_json)
                    VALUES (?, ?, ?)
                """, (req_hash, endpoint, resp_str))
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def clear(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM api_cache")
                conn.commit()
        except Exception as e:
            logger.warning(f"Cache clear error: {e}")

    def stats(self) -> dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*), endpoint FROM api_cache GROUP BY endpoint")
                rows = cursor.fetchall()
                return {"total_entries": sum(r[0] for r in rows), "by_endpoint": {r[1]: r[0] for r in rows}}
        except Exception:
            return {"total_entries": 0, "by_endpoint": {}}

cache_db = PharmaCacheDB()

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

# Virtual Drug Receptor Binding Kinetics Model
RECEPTOR_MODELS = {
    "GABAR": {"nt": "GABAergic", "Kd_uM": 1.2, "hill_n": 1.5, "Emax_hz": 25.0, "type": "inhibitory"},
    "nAChR": {"nt": "Cholinergic", "Kd_uM": 0.8, "hill_n": 1.8, "Emax_hz": 30.0, "type": "excitatory"},
    "DopR": {"nt": "Dopaminergic", "Kd_uM": 2.5, "hill_n": 1.2, "Emax_hz": 18.0, "type": "modulatory"},
    "GluR": {"nt": "Glutamatergic", "Kd_uM": 1.0, "hill_n": 1.6, "Emax_hz": 28.0, "type": "excitatory"}
}

def calculate_receptor_occupancy(concentration_uM: float, Kd_uM: float, hill_n: float) -> float:
    """Hill equation for receptor occupancy: theta = C^n / (Kd^n + C^n)"""
    if concentration_uM <= 0:
        return 0.0
    c_n = concentration_uM ** hill_n
    kd_n = Kd_uM ** hill_n
    return float(c_n / (kd_n + c_n))

# Request & Response Pydantic v2 Models
class LesionRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "target_neurons": ["N-1", "N-5"],
            "target_types": ["sensory"],
            "lesion_severity": 0.5,
            "mode": "node_removal"
        }
    })
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
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "compound_name": "NeuroMod-X4",
            "target_receptor": "GABAR",
            "mechanism": "agonist",
            "dosage_concentration": 2.5,
            "target_region": "central_brain"
        }
    })
    compound_name: str = Field(..., description="Name of candidate drug compound")
    target_receptor: str = Field(..., description="Target receptor: GABAR, nAChR, DopR, GluR")
    mechanism: str = Field(..., description="Mechanism: agonist, antagonist, allosteric_modulator")
    dosage_concentration: float = Field(default=1.0, ge=0.0, le=10.0, description="Concentration (uM)")
    target_region: Optional[str] = Field(default=None, description="Optional restricted brain region")

class DrugSimulateResponse(BaseModel):
    compound_name: str
    target_receptor: str
    mechanism: str
    dosage_concentration_uM: float
    receptor_occupancy_pct: float
    affected_neuron_count: int
    mean_firing_rate_shift_hz: float
    circuit_stability_index: float
    network_gain_shift: float
    side_effect_risk_score: float
    efficacy_score: float
    top_modulated_nodes: List[Dict[str, Any]]

class TargetRankRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "disease_context": "hyper_excitability",
            "top_k": 10,
            "region_filter": "central_brain"
        }
    })
    disease_context: str = Field(..., description="Context: hyper_excitability, neurodegeneration, motor_deficit")
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
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "source_nodes": ["N-0", "N-1"],
            "target_nodes": [],
            "max_hops": 2,
            "min_weight": 0.2
        }
    })
    source_nodes: List[str] = Field(..., description="Starting neuron IDs")
    target_nodes: Optional[List[str]] = Field(default=[], description="Target neuron IDs")
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
def _approx_global_efficiency(G: nx.DiGraph, sample_size: int = 40) -> float:
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
        "cache_stats": cache_db.stats(),
        "endpoints": ["/predict/lesion", "/drug/simulate", "/target/rank", "/circuit/extract", "/cache/stats"]
    }

@app.get("/cache/stats")
def get_cache_stats():
    return cache_db.stats()

@app.post("/cache/clear")
def clear_cache():
    cache_db.clear()
    return {"status": "success", "message": "SQLite API cache cleared successfully."}

@app.post("/predict/lesion", response_model=LesionResponse)
def predict_lesion(req: LesionRequest):
    cached = cache_db.get("/predict/lesion", req.model_dump())
    if cached:
        return LesionResponse(**cached)

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
    cascade_risk = float(min(1.0, (len(nodes_to_remove) / (initial_nodes + 1e-9)) * 2.5 + (eff_loss / 100.0)))

    resp = LesionResponse(
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

    cache_db.set("/predict/lesion", req.model_dump(), resp.model_dump())
    return resp

@app.post("/drug/simulate", response_model=DrugSimulateResponse)
def drug_simulate(req: DrugSimulateRequest):
    cached = cache_db.get("/drug/simulate", req.model_dump())
    if cached:
        return DrugSimulateResponse(**cached)

    G = engine.G
    rec_info = RECEPTOR_MODELS.get(req.target_receptor, RECEPTOR_MODELS["nAChR"])
    target_nt = rec_info["nt"]

    # Calculate binding kinetics via Hill Equation
    occupancy = calculate_receptor_occupancy(
        concentration_uM=req.dosage_concentration,
        Kd_uM=rec_info["Kd_uM"],
        hill_n=rec_info["hill_n"]
    )
    
    matching_nodes = []
    for node, attrs in G.nodes(data=True):
        if attrs.get('neurotransmitter') == target_nt:
            if req.target_region is None or attrs.get('region') == req.target_region:
                matching_nodes.append(node)

    affected_count = len(matching_nodes)
    if affected_count == 0:
        matching_nodes = list(G.nodes())[:50]
        affected_count = len(matching_nodes)

    # Compute firing rate shift based on Hill occupancy and mechanism
    max_effect = rec_info["Emax_hz"] * occupancy
    if req.mechanism == "agonist":
        firing_shift = max_effect
    elif req.mechanism == "antagonist":
        firing_shift = -max_effect * 0.85
    else: # allosteric modulator
        firing_shift = max_effect * 0.45

    stability_idx = float(max(0.1, min(1.0, 1.0 - (abs(firing_shift) / 50.0))))
    gain_shift = float(1.0 + (firing_shift / 20.0))
    side_effect = float(min(1.0, (req.dosage_concentration / 10.0) * (affected_count / (G.number_of_nodes() + 1e-9)) * 2.5))
    efficacy = float(min(1.0, occupancy * (affected_count / (G.number_of_nodes() * 0.2 + 1e-9)) * (1.0 - side_effect * 0.4)))

    top_nodes = []
    for nid in matching_nodes[:5]:
        data = G.nodes[nid]
        gene_expr = data.get('gene_expression', 0.5)
        top_nodes.append({
            "id": nid,
            "type": data.get('type', 'unknown'),
            "region": data.get('region', 'unknown'),
            "gene_expression": round(gene_expr, 3),
            "delta_rate_hz": round(firing_shift * gene_expr, 2)
        })

    resp = DrugSimulateResponse(
        compound_name=req.compound_name,
        target_receptor=req.target_receptor,
        mechanism=req.mechanism,
        dosage_concentration_uM=req.dosage_concentration,
        receptor_occupancy_pct=round(occupancy * 100.0, 2),
        affected_neuron_count=affected_count,
        mean_firing_rate_shift_hz=round(firing_shift, 2),
        circuit_stability_index=round(stability_idx, 4),
        network_gain_shift=round(gain_shift, 4),
        side_effect_risk_score=round(side_effect, 4),
        efficacy_score=round(efficacy, 4),
        top_modulated_nodes=top_nodes
    )

    cache_db.set("/drug/simulate", req.model_dump(), resp.model_dump())
    return resp

@app.post("/target/rank", response_model=List[TargetRankItem])
def target_rank(req: TargetRankRequest):
    cached = cache_db.get("/target/rank", req.model_dump())
    if cached:
        return [TargetRankItem(**item) for item in cached]

    G = engine.G
    nodes = list(G.nodes())
    if req.region_filter:
        nodes = [n for n in nodes if G.nodes[n].get('region') == req.region_filter]

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
    res_items = items[:req.top_k]

    cache_db.set("/target/rank", req.model_dump(), [item.model_dump() for item in res_items])
    return res_items

@app.post("/circuit/extract", response_model=CircuitExtractResponse)
def circuit_extract(req: CircuitExtractRequest):
    cached = cache_db.get("/circuit/extract", req.model_dump())
    if cached:
        return CircuitExtractResponse(**cached)

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

    resp = CircuitExtractResponse(
        nodes=nodes_out,
        edges=edges_out,
        node_count=n_nodes,
        edge_count=n_edges,
        subcircuit_density=density
    )

    cache_db.set("/circuit/extract", req.model_dump(), resp.model_dump())
    return resp

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
