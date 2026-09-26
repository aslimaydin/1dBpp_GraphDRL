# Deep Reinforcement Learning on Item-Compatibility Graphs for One-Dimensional Bin Packing

Trained GCN+PPO policy and inference code for the paper (arXiv:2609.25397, https://doi.org/10.48550/arXiv.2609.25397).

> **Note:** This repository contains the reference code for our submitted manuscript. The codebase is currently provided "as-is" for the review process. Full documentation, training scripts, and usage tutorials will be added upon the paper's acceptance.
## Requirements

Python 3.10+, PyTorch, NumPy:

```
pip install -r requirements.txt
```

## Run

```
python test.py                      # three built-in BPPLIB instances
python test.py my_instance.txt      # any BPPLIB-format file: n, C, then one weight per line
python test.py --beam 5 --seed 1    # beam width (default 5, as in the paper) and decoding seed (default 0)
```

`model.py` defines the GNN encoder and actor-critic network, `env.py` the item-compatibility-graph environment, `test.py` loads `model_weights.pth` and decodes with stochastic beam search, printing the bins found next to the known optimum and First-Fit Decreasing.
