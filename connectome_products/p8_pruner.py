"""
Product 8 Builder: Neuromorphic Edge-AI Pruner & Model Compressor
==================================================================

Applies fruit fly connectome biological sparsity constraints (80%+ sparsity)
to dense PyTorch neural network models (MLP, CNN) while preserving representation capacity.
"""

import os
import json
import logging
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger("Product8Builder")


def build_product_8(loader, output_dir: Path):
    """
    Builds and benchmarks Product 8: Neuromorphic Edge-AI Pruner.
    
    Args:
        loader: ConnectomeLoader instance.
        output_dir: Target directory path (8_edge_ai_pruner).
        
    Returns:
        dict: Pruning benchmark metrics and compression statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Product 8] Building Neuromorphic Edge-AI Pruner at {output_dir}")

    # Dynamically import pruner module
    pruner_path = output_dir / "pruner.py"
    spec = importlib.util.spec_from_file_location("pruner", pruner_path)
    pruner_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pruner_mod)

    BiologicalSparsityPruner = pruner_mod.BiologicalSparsityPruner

    # Instantiate Pruner with 85% target biological sparsity
    pruner = BiologicalSparsityPruner(target_sparsity=0.85, bio_topological_bias=0.25)

    # 1. Benchmark PyTorch MLP Model
    logger.info("  Pruning Multi-Layer Perceptron (MLP)...")
    mlp_dense = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    mlp_pruned = pruner.prune_model(mlp_dense, target_sparsity=0.85)

    sample_mlp_input = torch.randn(16, 256)
    mlp_benchmark = pruner.benchmark_compression(mlp_dense, mlp_pruned, sample_mlp_input)

    # 2. Benchmark PyTorch CNN Model
    logger.info("  Pruning Convolutional Neural Network (CNN)...")
    cnn_dense = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 10)
    )
    cnn_pruned = pruner.prune_model(cnn_dense, target_sparsity=0.80)

    sample_cnn_input = torch.randn(8, 3, 32, 32)
    cnn_benchmark = pruner.benchmark_compression(cnn_dense, cnn_pruned, sample_cnn_input)

    # Extract sparse COO tensors for edge deployment
    sparse_tensors_mlp = pruner.convert_to_sparse_tensors(mlp_pruned)

    # Save pruned model weights
    torch.save(mlp_pruned.state_dict(), output_dir / "pruned_mlp_state.pt")
    torch.save(cnn_pruned.state_dict(), output_dir / "pruned_cnn_state.pt")

    num_nodes = loader.G.number_of_nodes() if loader and hasattr(loader, 'G') and loader.G else 2000

    results = {
        "product_id": 8,
        "name": "Neuromorphic Edge-AI Pruner",
        "connectome_calibration_neurons": num_nodes,
        "mlp_pruning_benchmark": mlp_benchmark,
        "cnn_pruning_benchmark": cnn_benchmark,
        "sparse_tensors_generated": len(sparse_tensors_mlp),
        "status": "SUCCESS",
    }

    report_file = output_dir / "edge_ai_pruner_benchmark.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"✓ Product 8 build complete! Saved report to {report_file}")
    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from CONNECTOME_MASTER_ORCHESTRATOR import ConnectomeLoader
    loader = ConnectomeLoader().load()
    out = Path(__file__).parent / "8_edge_ai_pruner"
    build_product_8(loader, out)
