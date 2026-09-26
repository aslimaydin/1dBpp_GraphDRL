"""One-dimensional bin packing as a Markov decision process on the item-compatibility graph.

State    : undirected graph G_t whose nodes are the current partial bins (initially one node
           per item). An edge joins two nodes whose loads fit together in one bin
           (w_i + w_j <= C). Node features: [w_i / C, degree_i / max_degree].
Action   : choose an edge (i, j) the two nodes are merged into a single bin.
Terminal : no edge is left. The number of remaining nodes is the number of bins.
Reward   : 'step' (+1 per merge used by the released model), 'shaped', 'terminal'
           or 'utilization'.
"""
import random
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Graph construction and helpers
# ---------------------------------------------------------------------------

def build_graph(weights: List[int], capacity: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compatibility graph of the current node loads.

    Edge (i, j) exists iff weights[i] + weights[j] <= capacity.
    Returns (node_features (N, 2), adjacency (N, N) with zero diagonal).
    """
    n = len(weights)
    w_t = torch.tensor(weights, dtype=torch.float32)
    adj = (w_t.unsqueeze(0) + w_t.unsqueeze(1) <= capacity).float()
    adj.fill_diagonal_(0.0)

    degrees = adj.sum(dim=1)
    max_degree = max(degrees.max().item(), 1.0)

    node_features = torch.zeros(n, 2)
    node_features[:, 0] = w_t / capacity       # relative load w_i / C
    node_features[:, 1] = degrees / max_degree  # normalised degree
    return node_features, adj


def get_valid_edges(adj: torch.Tensor) -> torch.Tensor:
    """Upper-triangular edge list (E, 2) with i < j."""
    return torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=False)


def generate_random_instance(n_items: int, capacity: int, low: int = 1,
                             high: Optional[int] = None, seed: Optional[int] = None) -> List[int]:
    """Uniform random instance with integer weights in [low, high] (high defaults to C)."""
    if high is None:
        high = capacity
    rng = random.Random(seed) if seed is not None else random.Random()
    return [rng.randint(low, high) for _ in range(n_items)]


def first_fit_decreasing(items: List[int], capacity: int) -> Tuple[int, List[List[int]]]:
    """First-Fit Decreasing heuristic. Returns (number of bins, item indices per bin)."""
    indexed = sorted(enumerate(items), key=lambda x: x[1], reverse=True)
    bins_remaining: List[int] = []
    bins_contents: List[List[int]] = []
    for idx, weight in indexed:
        placed = False
        for b in range(len(bins_remaining)):
            if bins_remaining[b] >= weight:
                bins_remaining[b] -= weight
                bins_contents[b].append(idx)
                placed = True
                break
        if not placed:
            bins_remaining.append(capacity - weight)
            bins_contents.append([idx])
    return len(bins_contents), bins_contents


def load_bpplib_instance(path: str) -> Tuple[List[int], int]:
    """Reads a BPPLIB-format text file (line 1: n, line 2: C, then n weights)."""
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    n_items = int(lines[0])
    capacity = int(lines[1])
    weights = [int(lines[i]) for i in range(2, 2 + n_items)]
    return weights, capacity


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BinPackingGraphEnv:
    """Graph-based 1D bin packing environment.

    An episode starts with one node per item and its compatibility graph. Every step
    merges the two endpoints of the chosen edge and rebuilds the graph. The episode ends
    when no edge is left the remaining nodes are the bins.
    """

    def __init__(self, n_items: int = 50, capacity: int = 100,
                 reward_type: str = 'step', shaping_coef: float = 1.0):
        self.n_items = n_items
        self.capacity = capacity
        self.reward_type = reward_type
        self.shaping_coef = shaping_coef

        self.weights: List[int] = []        # current load of every node
        self.node_features = None           # (N, 2)
        self.adj = None                     # (N, N)
        self.done = False
        self.n_merges = 0
        self.initial_n_items = 0
        self.merge_history: List[Dict] = []
        self.node_contents: List[List[int]] = []   # original item indices in every node

    def reset(self, items: Optional[List[int]] = None,
              seed: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Starts a new episode generates a uniform random instance if items is None."""
        if items is None:
            items = generate_random_instance(self.n_items, self.capacity,
                                             low=1, high=self.capacity, seed=seed)
        self.weights = list(items)
        self.initial_n_items = len(items)
        self.done = False
        self.n_merges = 0
        self.merge_history = []
        self.node_contents = [[i] for i in range(len(items))]
        self._rebuild_graph()
        if len(get_valid_edges(self.adj)) == 0:
            self.done = True
        return self.get_state()

    def step(self, edge_idx: int) -> Tuple[Dict[str, torch.Tensor], float, bool, bool, Dict]:
        """Merges the two endpoints of valid edge number edge_idx.

        Returns (state, reward, done, truncated, info).
        """
        if self.done:
            raise RuntimeError("Episode is over call reset().")
        valid_edges = get_valid_edges(self.adj)
        if edge_idx < 0 or edge_idx >= len(valid_edges):
            raise ValueError(f"Invalid edge_idx={edge_idx}, valid range [0, {len(valid_edges) - 1}]")

        node_i = valid_edges[edge_idx][0].item()
        node_j = valid_edges[edge_idx][1].item()
        self.merge_history.append({
            'step': self.n_merges,
            'node_i': node_i,
            'node_j': node_j,
            'weight_i': self.weights[node_i],
            'weight_j': self.weights[node_j],
            'merged_weight': self.weights[node_i] + self.weights[node_j],
        })

        self._merge_nodes(node_i, node_j)
        self.n_merges += 1
        self._rebuild_graph()

        valid_edges_new = get_valid_edges(self.adj)
        if len(valid_edges_new) == 0:
            self.done = True

        reward = self._compute_reward()
        info = {
            'n_nodes': len(self.weights),
            'n_edges': len(valid_edges_new) if not self.done else 0,
            'n_merges': self.n_merges,
            'n_bins': len(self.weights),
        }
        return self.get_state(), reward, self.done, False, info

    def get_state(self) -> Dict[str, torch.Tensor]:
        return {
            'node_features': self.node_features.clone(),
            'adj': self.adj.clone(),
            'valid_edges': get_valid_edges(self.adj),
            'weights': list(self.weights),
            'n_nodes': len(self.weights),
        }

    def get_num_bins(self) -> int:
        """Current number of bins (= number of nodes)."""
        return len(self.weights)

    def _rebuild_graph(self):
        self.node_features, self.adj = build_graph(self.weights, self.capacity)

    def _merge_nodes(self, node_i: int, node_j: int):
        """Removes nodes i and j and appends the merged node at the end of the list."""
        if node_i > node_j:
            node_i, node_j = node_j, node_i
        new_weight = self.weights[node_i] + self.weights[node_j]
        new_contents = self.node_contents[node_i] + self.node_contents[node_j]
        del self.weights[node_j]
        del self.node_contents[node_j]
        del self.weights[node_i]
        del self.node_contents[node_i]
        self.weights.append(new_weight)
        self.node_contents.append(new_contents)

    def _compute_reward(self) -> float:
        if self.reward_type == 'step':
            # +1 per merge, 0 at the terminal step
            return 0.0 if self.done else 1.0

        elif self.reward_type == 'shaped':
            # +1 plus a potential-based term 2 w_i w_j / C^2 (in (0, 0.5])
            if self.done:
                return 0.0
            last = self.merge_history[-1]
            wi, wj = last['weight_i'], last['weight_j']
            return float(1.0 + self.shaping_coef * 2.0 * wi * wj / (self.capacity ** 2))

        elif self.reward_type == 'utilization':
            # squared fill ratio of the merged bin, +1 bonus if exactly full
            if self.done:
                return 0.0
            last = self.merge_history[-1]
            w_new = last['weight_i'] + last['weight_j']
            reward = (w_new / self.capacity) ** 2
            if w_new == self.capacity:
                reward += 1.0
            return float(reward)

        elif self.reward_type == 'terminal':
            # minus the number of bins at the terminal step, 0 otherwise
            return -float(len(self.weights)) if self.done else 0.0

        raise ValueError(f"Unknown reward type: {self.reward_type}")
