# Introduction

This is the official PyTorch implementation of the ICML 2023 paper titled "Bidirectional Adaptation for Robust Semi-Supervised Learning with Inconsistent Data Distributions."

#  Environment

Install conda environment via BidirectionalAdaptation.yaml

```
conda env create -f BidirectionalAdaptation.yaml
```

# Usage

For the basic dataset setup, run:

```python
python BDA.py --labels 300 --source c --target i --dataset image-CLEF --root data_path 
```

For the long-tailed dataset setup, run:

```
python BDA.py --labels 300 --source c --target i --dataset image-CLEF --root data_path --lt True
```

For the open-set dataset setup, run:

```
python BDA.py --labels 300 --source c --target i --dataset image-CLEF --root data_path --op True
```

The results will be printed and stored in `./results/`.
