#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File       : plot_embeddings.py
Author     : Ritesh
Created    : 2025-08-18
Description: <SHORT DESCRIPTION HERE>
"""

import yaml
def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)
config = load_config()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_embeddings(path="embeddings.npy", method="pca", title="Semantic Drift"):
    X = np.load(path)

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30)
    else:
        raise ValueError("Choose 'pca' or 'tsne'")

    X_reduced = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.6, s=20)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="embeddings.npy")
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--title", type=str, default="Semantic Drift")
    args = parser.parse_args()

    plot_embeddings(args.path, args.method, args.title)
