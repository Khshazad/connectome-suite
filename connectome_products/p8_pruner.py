"""
Product 8 Builder: Neuromorphic Edge-AI Model Pruner & Compressor
===================================================================

Applies fruit fly connectome biological log-normal graph sparsity (>85% target)
to dense PyTorch architectures (MLP, ConvNet, ResNet), converts pruned layers to COO/CSR sparse format,
and benchmarks compression metrics saved to `edge_ai_pruner_benchmark.json`.
"""

import os
import json
import logging
import importlib.util
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger("Product8Builder")


class MiniResNetBlock(nn.Module):
    """Miniature Residual Block with shortcut connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class MiniResNet(nn.Module):
    """Miniature Residual Neural Network for biological pruning benchmarks."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.layer1 = MiniResNetBlock(32)
        self.layer2 = MiniResNetBlock(32)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prep(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def build_product_8(loader, output_dir: Path):
    """
    Builds and benchmarks Product 8: Neuromorphic Edge-AI Model Pruner (>85% Sparsity).

    Args:
        loader: ConnectomeLoader instance.
        output_dir: Target directory path (8_edge_ai_pruner).

    Returns:
        dict: Detailed pruning benchmarks across MLP, CNN, and ResNet architectures.
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

    # Instantiate Biological Sparsity Pruner targeting >85% sparsity (0.88)
    pruner = BiologicalSparsityPruner(target_sparsity=0.88, bio_topological_bias=0.25)

    # 1. Benchmark Multi-Layer Perceptron (MLP)
    logger.info("  Pruning 1/3: Multi-Layer Perceptron (MLP)...")
    mlp_dense = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    mlp_pruned = pruner.prune_model(mlp_dense, target_sparsity=0.88)
    sample_mlp_input = torch.randn(16, 256)
    mlp_benchmark = pruner.benchmark_compression(mlp_dense, mlp_pruned, sample_mlp_input)
    sparse_tensors_mlp = pruner.convert_to_sparse_tensors(mlp_pruned)

    # 2. Benchmark Convolutional Neural Network (CNN)
    logger.info("  Pruning 2/3: Convolutional Neural Network (CNN)...")
    cnn_dense = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((8, 8)),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 10)
    )
    cnn_pruned = pruner.prune_model(cnn_dense, target_sparsity=0.86)
    sample_cnn_input = torch.randn(8, 3, 32, 32)
    cnn_benchmark = pruner.benchmark_compression(cnn_dense, cnn_pruned, sample_cnn_input)
    sparse_tensors_cnn = pruner.convert_to_sparse_tensors(cnn_pruned)

    # 3. Benchmark Residual Neural Network (ResNet)
    logger.info("  Pruning 3/3: Residual Neural Network (ResNet)...")
    resnet_dense = MiniResNet(in_channels=3, num_classes=10)
    resnet_pruned = pruner.prune_model(resnet_dense, target_sparsity=0.88)
    sample_resnet_input = torch.randn(8, 3, 32, 32)
    resnet_benchmark = pruner.benchmark_compression(resnet_dense, resnet_pruned, sample_resnet_input)
    sparse_tensors_resnet = pruner.convert_to_sparse_tensors(resnet_pruned)

    # Save pruned state dict model checkpoints
    torch.save(mlp_pruned.state_dict(), output_dir / "pruned_mlp_state.pt")
    torch.save(cnn_pruned.state_dict(), output_dir / "pruned_cnn_state.pt")
    torch.save(resnet_pruned.state_dict(), output_dir / "pruned_resnet_state.pt")

    num_nodes = loader.G.number_of_nodes() if loader and hasattr(loader, 'G') and loader.G else 2000

    results = {
        "product_id": 8,
        "name": "Neuromorphic Edge-AI Pruner & Model Compressor",
        "connectome_calibration_neurons": num_nodes,
        "target_biological_sparsity_pct": 88.0,
        "architectures_pruned": {
            "mlp": mlp_benchmark,
            "cnn": cnn_benchmark,
            "resnet": resnet_benchmark,
        },
        "coo_csr_sparse_conversions": {
            "mlp_sparse_layers": len(sparse_tensors_mlp),
            "cnn_sparse_layers": len(sparse_tensors_cnn),
            "resnet_sparse_layers": len(sparse_tensors_resnet),
            "formats_supported": ["COO", "CSR"],
        },
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
