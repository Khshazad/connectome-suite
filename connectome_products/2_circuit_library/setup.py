"""
Setup script for bio-inspired connectome circuit library PyPI package.
"""

from setuptools import setup, find_packages

setup(
    name="connectome-circuit-library",
    version="1.0.0",
    author="Connectome Suite Team",
    author_email="dev@connectome.ai",
    description="Bio-Inspired PyTorch LIF Spiking Neural Network Circuit SDK",
    long_description=open("README.md").read() if False else "Bio-inspired Drosophila SNN Circuit SDK",
    long_description_content_type="text/markdown",
    url="https://github.com/Khshazad/connectome-suite",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.20.0",
    ],
)
