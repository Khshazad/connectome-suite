#!/usr/bin/env python3
"""
Brain-Wide Gene-to-Circuit Receptor Mapper & Drug Susceptibility Engine
========================================================================
Integrates single-cell RNA-sequencing (scRNA-seq) neurotransmitter receptor profiles
(nAChR, GABAR, DopR, GluR) with connectomics graph topology to predict cell-type drug susceptibility,
vulnerability indices, and circuit pharmacology perturbation matrices.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("GeneCircuitMapper")

class GeneToCircuitMapper:
    """Cell-type & Receptor scRNA-seq Connectome Integration Engine."""

    RECEPTORS = [
        "nAChR_alpha1", "nAChR_beta2", "nAChR_alpha7",
        "GABAR_a1", "GABAR_b2", "GABAR_g2",
        "DopR_D1", "DopR_D2", "DopR_D3",
        "GluR_AMPA1", "GluR_NMDA1"
    ]

    RECEPTOR_FAMILIES = {
        "nAChR": ["nAChR_alpha1", "nAChR_beta2", "nAChR_alpha7"],
        "GABAR": ["GABAR_a1", "GABAR_b2", "GABAR_g2"],
        "DopR": ["DopR_D1", "DopR_D2", "DopR_D3"],
        "GluR": ["GluR_AMPA1", "GluR_NMDA1"]
    }

    def __init__(self, G: nx.DiGraph, neurons_df: pd.DataFrame):
        self.G = G
        self.neurons_df = neurons_df.copy()
        self.expression_matrix = None
        self._generate_scrna_profiles()

    def _generate_scrna_profiles(self):
        """Generate realistic scRNA-seq receptor expression matrix mapped to neuron types."""
        logger.info("Generating scRNA-seq receptor expression matrix for nAChR, GABAR, DopR, GluR...")
        np.random.seed(42)
        n_neurons = len(self.neurons_df)

        expr_data = {}
        for r in self.RECEPTORS:
            if "nAChR" in r:
                base = np.where(self.neurons_df['neurotransmitter'] == 'Cholinergic', 0.8, 0.2)
            elif "GABAR" in r:
                base = np.where(self.neurons_df['neurotransmitter'] == 'GABAergic', 0.85, 0.15)
            elif "DopR" in r:
                base = np.where(self.neurons_df['neurotransmitter'] == 'Dopaminergic', 0.9, 0.1)
            else:
                base = np.where(self.neurons_df['neurotransmitter'] == 'Glutamatergic', 0.75, 0.25)
            
            expr_data[r] = np.clip(base * np.random.lognormal(0.0, 0.4, n_neurons), 0.0, 5.0)

        self.expression_matrix = pd.DataFrame(expr_data, index=self.neurons_df['id'])
        
        # Annotate Graph Nodes with expression vectors
        for nid, row in self.expression_matrix.iterrows():
            if nid in self.G:
                self.G.nodes[nid]['receptor_expr'] = row.to_dict()

    def compute_celltype_susceptibility(self) -> pd.DataFrame:
        """Compute Cell-Type Drug Susceptibility & Vulnerability Scores."""
        logger.info("Computing cell-type drug susceptibility scores...")
        
        # Subsample graph for fast centrality computation
        nodes = list(self.G.nodes())
        sub_nodes = nodes[:150] if len(nodes) > 150 else nodes
        subG = self.G.subgraph(sub_nodes)
        
        bet_cent = nx.betweenness_centrality(subG, k=min(30, len(sub_nodes)))
        deg_cent = nx.degree_centrality(subG)

        self.neurons_df['betweenness'] = self.neurons_df['id'].map(lambda x: bet_cent.get(x, 0.0))
        self.neurons_df['degree'] = self.neurons_df['id'].map(lambda x: deg_cent.get(x, 0.0))

        # Group by neuron type & region
        type_groups = self.neurons_df.groupby(['type', 'region'])
        
        results = []
        for (ntype, region), group in type_groups:
            group_ids = group['id'].tolist()
            expr_sub = self.expression_matrix.loc[group_ids]
            
            total_expr_density = expr_sub.sum(axis=1).mean()
            mean_bet = group['betweenness'].mean()
            mean_deg = group['degree'].mean()

            # Formula: S = 0.4*ReceptorDensity + 0.35*Betweenness + 0.25*Degree
            susceptibility_score = float(0.4 * total_expr_density + 0.35 * mean_bet * 10.0 + 0.25 * mean_deg * 10.0)

            # Predominant receptor family
            family_means = {fam: expr_sub[[r for r in recs if r in expr_sub.columns]].values.mean() 
                            for fam, recs in self.RECEPTOR_FAMILIES.items()}
            top_fam = max(family_means, key=family_means.get)

            results.append({
                "cell_type": ntype,
                "brain_region": region,
                "neuron_count": len(group),
                "total_receptor_density": round(float(total_expr_density), 4),
                "mean_betweenness": round(float(mean_bet), 4),
                "mean_degree": round(float(mean_deg), 4),
                "susceptibility_score": round(susceptibility_score, 4),
                "primary_target_family": top_fam,
                "family_expression": {k: round(float(v), 3) for k, v in family_means.items()}
            })

        res_df = pd.DataFrame(results).sort_values("susceptibility_score", ascending=False)
        return res_df

    def simulate_pharmacology_matrix(self) -> Dict[str, Any]:
        """Generate receptor x cell-type drug response perturbation matrix."""
        logger.info("Simulating receptor pharmacology perturbation matrix...")
        cell_types = self.neurons_df['type'].unique().tolist()
        
        matrix = {}
        for fam in self.RECEPTOR_FAMILIES.keys():
            matrix[fam] = {}
            for ct in cell_types:
                ct_ids = self.neurons_df[self.neurons_df['type'] == ct]['id'].tolist()
                recs = self.RECEPTOR_FAMILIES[fam]
                expr_val = self.expression_matrix.loc[ct_ids, recs].values.mean() if ct_ids else 0.0
                
                response_val = float(np.tanh(expr_val * 0.8) * 100.0)
                matrix[fam][ct] = round(response_val, 2)

        return {
            "receptor_families": list(self.RECEPTOR_FAMILIES.keys()),
            "cell_types": cell_types,
            "perturbation_matrix_pct": matrix
        }

    def export_all(self, output_dir: Path) -> Dict[str, str]:
        """Export CSV and JSON reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        susc_df = self.compute_celltype_susceptibility()
        csv_path = output_dir / "cell_type_susceptibility_ranking.csv"
        susc_df.to_csv(csv_path, index=False)

        json_susc_path = output_dir / "cell_type_susceptibility_ranking.json"
        with open(json_susc_path, "w") as f:
            json.dump(susc_df.to_dict(orient="records"), f, indent=2)

        pharm_matrix = self.simulate_pharmacology_matrix()
        pharm_path = output_dir / "receptor_pharmacology_matrix.json"
        with open(pharm_path, "w") as f:
            json.dump(pharm_matrix, f, indent=2)

        expr_summary_path = output_dir / "gene_expression_summary.json"
        expr_summary = {
            "receptors": self.RECEPTORS,
            "mean_expression": self.expression_matrix.mean().to_dict(),
            "max_expression": self.expression_matrix.max().to_dict()
        }
        with open(expr_summary_path, "w") as f:
            json.dump(expr_summary, f, indent=2)

        logger.info(f"✓ Created Gene-to-Circuit Mapper reports in {output_dir}")
        return {
            "csv_path": str(csv_path),
            "json_susc_path": str(json_susc_path),
            "pharm_path": str(pharm_path),
            "expr_summary_path": str(expr_summary_path)
        }

if __name__ == "__main__":
    print("Gene-to-Circuit Mapper Engine Loaded.")
