"""GNN encoder and actor-critic network for bin packing on the item-compatibility graph.

Graph encoder (GCN, GAT or GIN layers, residual + LayerNorm) -> node embeddings h_i
State aggregator (mean over nodes)                           -> state vector
Policy (actor) : score every feasible edge from [h_i || h_j], softmax over edges
Value  (critic): V(s) from the state vector
Q-network      : Q(s, a) for the value-based variants (DQN, SAC, SARSA)

The released model is GCN + PPO: embed_dim 128, 3 layers, mean aggregation.
"""
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GNN layers (plain PyTorch; all share forward(x, adj) -> x')
# ---------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """Graph convolution (Kipf & Welling): h' = D^-1/2 (A + I) D^-1/2 h W."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.weight)
        if self.weight.bias is not None:
            nn.init.zeros_(self.weight.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = adj.size(0)
        adj_hat = adj + torch.eye(N, device=adj.device)
        degree = adj_hat.sum(dim=1).clamp(min=1)
        D_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
        adj_norm = D_inv_sqrt @ adj_hat @ D_inv_sqrt
        return adj_norm @ self.weight(x)


class GATLayer(nn.Module):
    """Multi-head graph attention (Velickovic et al.)."""

    def __init__(self, in_features: int, out_features: int, n_heads: int = 4,
                 concat: bool = True, dropout: float = 0.1, negative_slope: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.W = nn.Parameter(torch.Tensor(n_heads, in_features, out_features))
        self.a_left = nn.Parameter(torch.Tensor(n_heads, out_features, 1))
        self.a_right = nn.Parameter(torch.Tensor(n_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.attn_dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.Tensor(n_heads * out_features if concat else out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        adj_hat = adj + torch.eye(N, device=adj.device)
        Wh = torch.einsum('ni,kio->kno', x, self.W)          # (K, N, d_out)
        e_left = torch.bmm(Wh, self.a_left)                   # (K, N, 1)
        e_right = torch.bmm(Wh, self.a_right)                 # (K, N, 1)
        e = self.leaky_relu(e_left + e_right.transpose(1, 2))  # (K, N, N)
        mask = adj_hat.unsqueeze(0).expand(self.n_heads, -1, -1)
        e = e.masked_fill(mask == 0, float('-inf'))
        alpha = self.attn_dropout(F.softmax(e, dim=2))
        h_prime = torch.bmm(alpha, Wh)                        # (K, N, d_out)
        if self.concat:
            out = h_prime.permute(1, 0, 2).contiguous().view(N, -1)
        else:
            out = h_prime.mean(dim=0)
        return out + self.bias


class GINLayer(nn.Module):
    """Graph isomorphism layer """

    def __init__(self, in_features: int, out_features: int,
                 hidden_features: Optional[int] = None, eps_learnable: bool = True):
        super().__init__()
        if hidden_features is None:
            hidden_features = out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
        )
        if eps_learnable:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('eps', torch.zeros(1))
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return self.mlp((1 + self.eps) * x + adj @ x)


def get_gnn_layer(gnn_type: str, in_features: int, out_features: int,
                  n_heads: int = 4, **kwargs) -> nn.Module:
    gnn_type = gnn_type.lower()
    if gnn_type == 'gcn':
        return GCNLayer(in_features, out_features)
    if gnn_type == 'gat':
        return GATLayer(in_features, out_features, n_heads=n_heads,
                        concat=kwargs.get('concat', True), dropout=kwargs.get('dropout', 0.1))
    if gnn_type == 'gin':
        return GINLayer(in_features, out_features,
                        hidden_features=kwargs.get('hidden_features', out_features))
    raise ValueError(f"Unknown GNN type: {gnn_type} (expected gcn, gat or gin)")


# ---------------------------------------------------------------------------
# Encoder, aggregator, actor, critic
# ---------------------------------------------------------------------------

class GraphEncoder(nn.Module):

    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 128, n_layers: int = 3,
                 gnn_type: str = 'gat', n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.gnn_type = gnn_type
        self.input_embedding = nn.Sequential(nn.Linear(node_feat_dim, embed_dim), nn.ReLU())
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for l in range(n_layers):
            if gnn_type == 'gat':
                if l == n_layers - 1:      # last layer: average the heads
                    layer = get_gnn_layer('gat', embed_dim, embed_dim, n_heads=n_heads,
                                          concat=False, dropout=dropout)
                else:                      # concatenate heads: n_heads * (embed_dim // n_heads)
                    layer = get_gnn_layer('gat', embed_dim, embed_dim // n_heads, n_heads=n_heads,
                                          concat=True, dropout=dropout)
            else:
                layer = get_gnn_layer(gnn_type, embed_dim, embed_dim, n_heads=n_heads)
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(embed_dim))

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_embedding(node_features)
        for gnn_layer, layer_norm in zip(self.gnn_layers, self.layer_norms):
            h_new = self.dropout(F.relu(gnn_layer(h, adj)))
            h = layer_norm(h + h_new)
        return h


class StateAggregator(nn.Module):
    """Fixed-size state vector ('sum', 'mean', 'max' or 'mlp')."""

    def __init__(self, embed_dim: int = 128, agg_type: str = 'mean'):
        super().__init__()
        self.agg_type = agg_type
        self.embed_dim = embed_dim
        if agg_type == 'mlp':
            self.mlp = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(),
                                     nn.Linear(embed_dim, embed_dim))

    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        if self.agg_type == 'sum':
            return node_embeddings.sum(dim=0)
        if self.agg_type == 'mean':
            return node_embeddings.mean(dim=0)
        if self.agg_type == 'max':
            return node_embeddings.max(dim=0)[0]
        if self.agg_type == 'mlp':
            return self.mlp(node_embeddings.mean(dim=0))
        raise ValueError(f"Unknown aggregation type: {self.agg_type}")


class PolicyNetwork(nn.Module):

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.score_network = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, node_embeddings: torch.Tensor, valid_edges: torch.Tensor) -> torch.Tensor:
        if len(valid_edges) == 0:
            return torch.tensor([])
        action_vectors = self.get_action_embeddings(node_embeddings, valid_edges)
        scores = self.score_network(action_vectors).squeeze(-1)
        return F.log_softmax(scores, dim=0)

    def get_action_embeddings(self, node_embeddings: torch.Tensor,
                              valid_edges: torch.Tensor) -> torch.Tensor:
        h_i = node_embeddings[valid_edges[:, 0]]
        h_j = node_embeddings[valid_edges[:, 1]]
        return torch.cat([h_i, h_j], dim=1)


class ValueNetwork(nn.Module):

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_vector: torch.Tensor) -> torch.Tensor:
        return self.value_net(state_vector).squeeze(-1)


class QNetwork(nn.Module):

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.q_net = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state_vector: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        if action_embedding.dim() == 1:
            return self.q_net(torch.cat([state_vector, action_embedding], dim=0)).squeeze(-1)
        state_expanded = state_vector.unsqueeze(0).expand(action_embedding.size(0), -1)
        return self.q_net(torch.cat([state_expanded, action_embedding], dim=1)).squeeze(-1)


class BPPActorCritic(nn.Module):

    def __init__(self, node_feat_dim: int = 2, embed_dim: int = 128, n_gnn_layers: int = 3,
                 gnn_type: str = 'gat', n_heads: int = 4, agg_type: str = 'mean',
                 policy_hidden: int = 128, value_hidden: int = 128, dropout: float = 0.1,
                 use_q_network: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.use_q_network = use_q_network
        self.encoder = GraphEncoder(node_feat_dim=node_feat_dim, embed_dim=embed_dim,
                                    n_layers=n_gnn_layers, gnn_type=gnn_type,
                                    n_heads=n_heads, dropout=dropout)
        self.aggregator = StateAggregator(embed_dim=embed_dim, agg_type=agg_type)
        self.policy = PolicyNetwork(embed_dim=embed_dim, hidden_dim=policy_hidden)
        self.value = ValueNetwork(embed_dim=embed_dim, hidden_dim=value_hidden)
        if use_q_network:
            self.q_net = QNetwork(embed_dim=embed_dim, hidden_dim=value_hidden)

    @property
    def device(self):
        return next(self.parameters()).device

    def encode(self, node_features: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embeddings = self.encoder(node_features, adj)
        return node_embeddings, self.aggregator(node_embeddings)

    def select_action(self, state: Dict[str, torch.Tensor],
                      greedy: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:

        dev = self.device
        node_features = state['node_features'].to(dev)
        adj = state['adj'].to(dev)
        valid_edges = state['valid_edges'].to(dev)
        if len(valid_edges) == 0:
            return -1, torch.tensor(0.0, device=dev), torch.tensor(0.0, device=dev)
        node_embeddings, state_vector = self.encode(node_features, adj)
        log_probs = self.policy(node_embeddings, valid_edges)
        value = self.value(state_vector)
        if greedy:
            edge_idx = torch.argmax(log_probs).item()
        else:
            edge_idx = torch.distributions.Categorical(torch.exp(log_probs)).sample().item()
        return edge_idx, log_probs[edge_idx], value

    def evaluate_action(self, state: Dict[str, torch.Tensor],
                        edge_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dev = self.device
        node_embeddings, state_vector = self.encode(state['node_features'].to(dev), state['adj'].to(dev))
        log_probs = self.policy(node_embeddings, state['valid_edges'].to(dev))
        value = self.value(state_vector)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum()
        return log_probs[edge_idx], value, entropy

    def get_q_values(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if not self.use_q_network:
            raise RuntimeError("Q-network not present; create the model with use_q_network=True.")
        dev = self.device
        valid_edges = state['valid_edges'].to(dev)
        node_embeddings, state_vector = self.encode(state['node_features'].to(dev), state['adj'].to(dev))
        action_embeddings = self.policy.get_action_embeddings(node_embeddings, valid_edges)
        return self.q_net(state_vector, action_embeddings), valid_edges

    def solve_greedy(self, env) -> Tuple[int, List]:
        """Runs the greedy (argmax) policy on an environment that has been reset."""
        state = env.get_state()
        merge_history = []
        with torch.no_grad():
            while not env.done:
                edge_idx, _, _ = self.select_action(state, greedy=True)
                if edge_idx < 0:
                    break
                state, _, _, _, info = env.step(edge_idx)
                merge_history.append(info)
        return env.get_num_bins(), merge_history


# ---------------------------------------------------------------------------
# Released model
# ---------------------------------------------------------------------------

def load_pretrained(weights_path: str = None, device: str = 'cpu') -> BPPActorCritic:
    """Builds the paper's GCN+PPO model and loads model_weights.pth."""
    if weights_path is None:
        weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_weights.pth')
    model = BPPActorCritic(node_feat_dim=2, embed_dim=128, n_gnn_layers=3, gnn_type='gcn',
                           n_heads=4, agg_type='mean', policy_hidden=128, value_hidden=128,
                           dropout=0.1, use_q_network=False)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model
