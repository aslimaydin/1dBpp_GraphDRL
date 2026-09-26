"""Runs the trained model on BPPLIB instances. No training involved.

    python test.py                        # three built-in BPPLIB instances
    python test.py path/to/instance.txt   # any file in BPPLIB format (n, C, then n weights)
    python test.py --beam 5 --seed 1      # beam width (default 5, as in the paper) and decoding seed

For every instance it prints the best known optimum, the bins used by First-Fit
Decreasing, by the model with greedy decoding, and by the model with stochastic
beam search (the decoder used in the paper, B = 5).
"""
import argparse
import os
import time
from typing import List

import torch

from env import BinPackingGraphEnv, first_fit_decreasing, load_bpplib_instance
from model import load_pretrained

# Built-in instances from BPPLIB (Delorme, Iori & Martello, Optimization Letters 2018;
# https://github.com/mdelorme2/BPPLIB, CC BY-NC-ND 4.0): (name, capacity, optimum, weights)
INSTANCES = [
    ('Falkenauer_t60_00 (Falkenauer T)', 1000, 20,
     [495, 474, 473, 472, 466, 450, 445, 444, 439, 430, 419, 414, 410, 395, 372, 370, 366, 366, 366,
      363, 361, 357, 355, 351, 350, 350, 347, 320, 315, 307, 303, 299, 298, 298, 292, 288, 287, 283,
      275, 275, 274, 273, 273, 272, 272, 271, 269, 269, 268, 263, 262, 261, 259, 258, 255, 254, 252,
      252, 252, 251]),
    ('N1W1B1R0 (Scholl 2)', 1000, 18,
     [395, 394, 394, 391, 390, 389, 388, 384, 383, 382, 380, 379, 376, 371, 368, 365, 360, 360, 354,
      350, 346, 346, 344, 342, 340, 335, 335, 333, 330, 330, 328, 327, 317, 316, 311, 310, 310, 306,
      300, 300, 297, 296, 295, 294, 294, 286, 285, 278, 275, 275]),
    ('Schwerin1_BPP1 (Schwerin 1)', 1000, 18,
     [200, 200, 200, 199, 198, 198, 197, 197, 194, 194, 193, 192, 191, 191, 191, 190, 190, 189, 188,
      188, 187, 187, 186, 185, 185, 185, 185, 184, 184, 184, 183, 183, 183, 182, 182, 182, 181, 181,
      180, 179, 179, 179, 179, 178, 177, 177, 177, 177, 175, 174, 173, 173, 172, 171, 171, 171, 170,
      170, 169, 169, 169, 167, 167, 165, 165, 164, 163, 163, 163, 163, 162, 161, 160, 160, 159, 158,
      158, 158, 157, 156, 156, 156, 156, 156, 156, 155, 155, 155, 154, 154, 153, 152, 152, 152, 151,
      151, 150, 150, 150, 150]),
]


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

def solve_greedy(model, weights: List[int], capacity: int) -> int:
    """Argmax policy from the initial graph to the terminal state."""
    env = BinPackingGraphEnv(n_items=len(weights), capacity=capacity, reward_type='step')
    env.reset(items=weights)
    return model.solve_greedy(env)[0]


class _Beam:
    __slots__ = ['env', 'score', 'done']

    def __init__(self, env: BinPackingGraphEnv, score: float = 0.0):
        self.env = env
        self.score = score
        self.done = env.done

    def clone(self) -> '_Beam':
        e = BinPackingGraphEnv(n_items=self.env.initial_n_items, capacity=self.env.capacity,
                               reward_type=self.env.reward_type)
        e.weights = list(self.env.weights)
        e.node_contents = [list(c) for c in self.env.node_contents]
        e.done = self.env.done
        e.n_merges = self.env.n_merges
        e.initial_n_items = self.env.initial_n_items
        e.merge_history = list(self.env.merge_history)
        e._rebuild_graph()
        return _Beam(e, self.score)


def solve_stochastic_beam(model, weights: List[int], capacity: int,
                          beam_width: int = 5, device: str = 'cpu') -> int:
    """Stochastic beam search: at every step each beam samples B distinct edges from the
    policy."""
    env0 = BinPackingGraphEnv(n_items=len(weights), capacity=capacity, reward_type='step')
    env0.reset(items=weights)
    beams = [_Beam(env0, score=0.0)]

    with torch.no_grad():
        while True:
            active = [b for b in beams if not b.done]
            if not active:
                break
            candidates = []
            for beam in active:
                state = beam.env.get_state()
                valid_edges = state['valid_edges'].to(device)
                if len(valid_edges) == 0:
                    beam.done = True
                    candidates.append(beam)
                    continue
                node_embeddings, _ = model.encode(state['node_features'].to(device), state['adj'].to(device))
                log_probs = model.policy(node_embeddings, valid_edges)
                probs = torch.exp(log_probs)
                k = min(beam_width, len(valid_edges))
                sampled = torch.multinomial(probs, k, replacement=False)
                for i in range(k):
                    edge_idx = sampled[i].item()
                    new_beam = beam.clone()
                    new_beam.score += log_probs[edge_idx].item()
                    try:
                        new_beam.env.step(edge_idx)
                        new_beam.done = new_beam.env.done
                    except Exception:
                        new_beam.done = True
                    candidates.append(new_beam)
            done_beams = [b for b in candidates if b.done]
            active_beams = [b for b in candidates if not b.done]
            active_beams.sort(key=lambda b: (b.env.get_num_bins(), -b.score))
            active_beams = active_beams[:beam_width]
            beams = active_beams + done_beams
            if not active_beams:
                break
    return min(b.env.get_num_bins() for b in beams)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('instance', nargs='*', help='BPPLIB-format instance file(s); default: built-in instances')
    ap.add_argument('--weights', default=None, help='path to model_weights.pth (default: next to this script)')
    ap.add_argument('--beam', type=int, default=5, help='beam width (default 5)')
    ap.add_argument('--seed', type=int, default=0, help='decoding seed (default 0)')
    args = ap.parse_args()

    model = load_pretrained(args.weights)

    if args.instance:
        cases = []
        for path in args.instance:
            weights, capacity = load_bpplib_instance(path)
            cases.append((os.path.basename(path), capacity, None, weights))
    else:
        cases = INSTANCES

    print(f"{'instance':34} {'n':>4} {'C':>6} {'opt':>4} {'FFD':>4} {'greedy':>6} {'beam':>5}   time")
    for name, capacity, opt, weights in cases:
        t0 = time.time()
        ffd = first_fit_decreasing(weights, capacity)[0]
        greedy = solve_greedy(model, weights, capacity)
        torch.manual_seed(args.seed)
        beam = solve_stochastic_beam(model, weights, capacity, beam_width=args.beam)
        print(f"{name:34} {len(weights):4d} {capacity:6d} {str(opt) if opt is not None else '?':>4} "
              f"{ffd:4d} {greedy:6d} {beam:5d}   {time.time() - t0:4.1f}s")


if __name__ == '__main__':
    main()
