# Feature Clustering

## Introduction

Toolbox designed to aid with the selection of features. Check github wiki to more details about the implemented methodologies.

## Build Package

1. Install build to your python environment:

```
pip install build>=0.8.0
```

2. Build package

```
python -m build
```

3. Check generated packages under `dist/` directory.

## Installing package

1. Run pip install with desired built package. Replace `[DESIRED_VERSION]` with desired version number. Example: 0.0.2

```
pip install ./dist/feature_clustering-[DESIRED_VERSION]-py3-none-any.whl 
```

## Usage

Basic usage:

```python
import pandas as pd
from feature_clustering.feature_clustering import FeatureCluster

# Load dataset
data = pd.read_csv("../data/datos_income_TFM.csv")
data.head()

# Select variable
target = "DIABETE4"

# Fit KMeans on desired X,Y with desired k value
feature_cluster = FeatureCluster(k=8)
feature_cluster.fit(data.drop(columns=target), data[target])

# Plot segmentation with umap
feature_cluster.plot(plot_type="umap")
>>> <figure>
```
Check demos on more details on how to use the package.
