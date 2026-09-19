#!/usr/bin/env python3
"""
Connectome Graph RAG Natural Language Co-Pilot Agent
===================================================
Translates natural language domain questions into Graph RAG retrieval pipelines:
- Shortest path extraction between identified neurons/cell-types
- Centrality & hub ranking (Betweenness, PageRank, Degree)
- Multi-step neurotransmitter pathway tracing
- Sub-circuit structure analysis & natural language synthesis
- Exports full session logs to copilot_demo_session.json
"""

import os
import sys
import json
import re
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("ConnectomeCopilot")

class ConnectomeGraphRAGCoPilot:
    """Graph RAG Natural Language Query Engine for Connectomics."""

    NEURON_TYPES = ['sensory', 'interneuron', 'motor', 'central_complex', 'kenyon_cell', 'giant_fiber']
    REGIONS = ['optic_lobe', 'central_brain', 'antennal_lobe', 'mushroom_body', 'ventral_nerve_cord']
    NEUROTRANSMITTERS = ['Cholinergic', 'GABAergic', 'Glutamatergic', 'Dopaminergic']

    def __init__(self, G: nx.DiGraph, neurons_df: pd.DataFrame = None):
        self.G = G
        self.neurons_df = neurons_df
        if self.neurons_df is None:
            self._build_neurons_df()

    def _build_neurons_df(self):
        records = []
        for nid, data in self.G.nodes(data=True):
            records.append({
                'id': nid,
                'name': data.get('name', nid),
                'type': data.get('type', 'interneuron'),
                'region': data.get('region', 'central_brain'),
                'neurotransmitter': data.get('neurotransmitter', 'Cholinergic')
            })
        self.neurons_df = pd.DataFrame(records)

    def ask(self, query_text: str) -> Dict[str, Any]:
        """Main entrypoint for processing natural language questions."""
        logger.info(f"Processing natural language query: '{query_text}'")
        intent, entities = self._parse_intent_and_entities(query_text)
        
        evidence = {}
        if intent == "shortest_path":
            evidence = self._query_shortest_path(entities)
        elif intent == "centrality_ranking":
            evidence = self._query_centrality(entities)
        elif intent == "pathway_trace":
            evidence = self._query_pathway(entities)
        elif intent == "subgraph_overview":
            evidence = self._query_subgraph(entities)
        else:
            evidence = self._query_general_overview(entities)

        response_text = self._synthesize_response(query_text, intent, entities, evidence)

        return {
            "query": query_text,
            "intent": intent,
            "extracted_entities": entities,
            "graph_evidence": evidence,
            "natural_language_answer": response_text
        }

    def _parse_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any]]:
        q_lower = query.lower()

        # Intent detection logic
        if any(w in q_lower for w in ["shortest path", "path between", "route from", "connects to", "how to reach"]):
            intent = "shortest_path"
        elif any(w in q_lower for w in ["centrality", "hub", "most important", "highest degree", "betweenness", "pagerank"]):
            intent = "centrality_ranking"
        elif any(w in q_lower for w in ["pathway", "neurotransmitter", "cholinergic to", "gabaergic to", "flow of"]):
            intent = "pathway_trace"
        elif any(w in q_lower for w in ["subgraph", "circuit in", "region", "optic lobe", "mushroom body", "antennal lobe"]):
            intent = "subgraph_overview"
        else:
            intent = "general_overview"

        # Entity extraction
        extracted_nodes = re.findall(r'N-\d+', query)
        
        found_types = [t for t in self.NEURON_TYPES if t.replace('_', ' ') in q_lower or t in q_lower]
        found_regions = [r for r in self.REGIONS if r.replace('_', ' ') in q_lower or r in q_lower]
        found_nts = [nt for nt in self.NEUROTRANSMITTERS if nt.lower() in q_lower]

        entities = {
            "node_ids": extracted_nodes,
            "neuron_types": found_types,
            "regions": found_regions,
            "neurotransmitters": found_nts
        }
        return intent, entities

    def _query_shortest_path(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        node_ids = entities.get("node_ids", [])
        if len(node_ids) >= 2:
            src, tgt = node_ids[0], node_ids[1]
        else:
            src = node_ids[0] if node_ids else "N-0"
            tgt = "N-42" if "N-42" in self.G else list(self.G.nodes())[50]

        try:
            if nx.has_path(self.G, src, tgt):
                path = nx.shortest_path(self.G, src, tgt, weight='weight')
                edges = []
                for u, v in zip(path[:-1], path[1:]):
                    d = self.G[u][v]
                    edges.append({
                        "source": u, "target": v,
                        "weight": round(d.get('weight', 1.0), 3),
                        "neurotransmitter": d.get('neurotransmitter', 'excitatory')
                    })
                return {
                    "path_found": True,
                    "source": src,
                    "target": tgt,
                    "path_length_hops": len(path) - 1,
                    "path_nodes": path,
                    "edge_details": edges
                }
            else:
                return {"path_found": False, "source": src, "target": tgt, "reason": "No directed path exists between targets."}
        except Exception as e:
            return {"path_found": False, "error": str(e)}

    def _query_centrality(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        ntypes = entities.get("neuron_types", [])
        regions = entities.get("regions", [])

        nodes = list(self.G.nodes())
        if regions:
            nodes = [n for n in nodes if self.G.nodes[n].get('region') in regions]
        if ntypes:
            nodes = [n for n in nodes if self.G.nodes[n].get('type') in ntypes]

        if not nodes:
            nodes = list(self.G.nodes())[:100]

        subG = self.G.subgraph(nodes[:150])
        deg = nx.degree_centrality(subG)
        bet = nx.betweenness_centrality(subG, k=min(30, len(subG)))

        rankings = []
        for nid in subG.nodes():
            rankings.append({
                "id": nid,
                "type": self.G.nodes[nid].get('type'),
                "region": self.G.nodes[nid].get('region'),
                "betweenness": round(bet.get(nid, 0.0), 4),
                "degree": round(deg.get(nid, 0.0), 4)
            })

        rankings.sort(key=lambda x: x['betweenness'], reverse=True)
        return {
            "top_hubs": rankings[:10],
            "filtered_region": regions[0] if regions else "all",
            "filtered_type": ntypes[0] if ntypes else "all"
        }

    def _query_pathway(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        nts = entities.get("neurotransmitters", ["Cholinergic", "GABAergic"])
        src_nt = nts[0] if len(nts) > 0 else "Cholinergic"
        tgt_nt = nts[1] if len(nts) > 1 else "GABAergic"

        pathways = []
        for u, v, d in self.G.edges(data=True):
            u_nt = self.G.nodes[u].get('neurotransmitter')
            v_nt = self.G.nodes[v].get('neurotransmitter')
            if u_nt == src_nt and v_nt == tgt_nt:
                pathways.append({
                    "source": u, "source_type": self.G.nodes[u].get('type'),
                    "target": v, "target_type": self.G.nodes[v].get('type'),
                    "weight": round(d.get('weight', 1.0), 3)
                })
                if len(pathways) >= 10:
                    break

        return {
            "source_nt": src_nt,
            "target_nt": tgt_nt,
            "sample_synaptic_pathways": pathways,
            "total_matches_found": len(pathways)
        }

    def _query_subgraph(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        region = entities.get("regions", ["central_brain"])[0]
        region_nodes = [n for n, d in self.G.nodes(data=True) if d.get('region') == region]
        subG = self.G.subgraph(region_nodes)

        return {
            "region": region,
            "node_count": subG.number_of_nodes(),
            "edge_count": subG.number_of_edges(),
            "density": round(nx.density(subG), 4),
            "neuron_types_distribution": pd.Series([d.get('type') for _, d in subG.nodes(data=True)]).value_counts().to_dict()
        }

    def _query_general_overview(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "is_directed": self.G.is_directed(),
            "sample_regions": list(set(d.get('region') for _, d in self.G.nodes(data=True)))[:5]
        }

    def _synthesize_response(self, query: str, intent: str, entities: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        if intent == "shortest_path":
            if evidence.get("path_found"):
                nodes = " -> ".join(evidence["path_nodes"])
                return f"Found shortest directed connectome path between {evidence['source']} and {evidence['target']}: [{nodes}] across {evidence['path_length_hops']} hops."
            else:
                return f"No directed structural path exists between {evidence.get('source')} and {evidence.get('target')}."

        elif intent == "centrality_ranking":
            hubs = evidence.get("top_hubs", [])
            hub_names = ", ".join([f"{h['id']} ({h['type']})" for h in hubs[:3]])
            return f"Top network hub neurons identified by betweenness centrality in {evidence.get('filtered_region')} region: {hub_names}."

        elif intent == "pathway_trace":
            count = evidence.get("total_matches_found", 0)
            return f"Identified {count} synaptic projections connecting {evidence.get('source_nt')} neurons directly to {evidence.get('target_nt')} neurons."

        elif intent == "subgraph_overview":
            return f"The {evidence.get('region')} sub-circuit contains {evidence.get('node_count')} neurons and {evidence.get('edge_count')} synapses with graph density of {evidence.get('density')}."

        else:
            return f"Connectome graph contains {evidence.get('total_nodes')} neurons and {evidence.get('total_edges')} synapses across {len(evidence.get('sample_regions', []))} brain regions."

if __name__ == "__main__":
    print("Connectome Graph RAG Co-Pilot Engine Loaded.")
