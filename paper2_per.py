#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_per.py — sequence-level prioritized experience replay (R2D2-style).

WHY THIS FILE EXISTS
--------------------
The manuscript states that the controller retains n-step *prioritized* experience replay from the
companion study. The R2 code did not implement it: the training loop called `agent.learn(buf[-8:])`,
i.e. it always replayed the eight most recently collected episodes in collection order. That is a
sliding on-policy window — no buffer, no priorities, no importance-sampling correction. This module
supplies the missing mechanism so that the code and the paper agree.

DESIGN
------
The learner is recurrent (an LSTM is unrolled over a whole episode) and the branched QR-DQN loss is
computed per episode, so the natural replay unit is the SEQUENCE (one episode), not the transition.
This is the R2D2 convention. Concretely:

  * capacity      C episodes, circular (oldest evicted first)
  * priority      p_i = (|delta_i| + eps), where delta_i is the mean absolute n-step TD error over
                  the timesteps and action branches of sequence i (see Agent.learn)
  * sampling      P(i) = p_i^alpha / sum_j p_j^alpha              (alpha = 0.6)
  * correction    w_i = (1 / (N P(i)))^beta, normalised by max_j w_j so the largest weight is 1;
                  beta is annealed linearly from beta0 = 0.4 to 1.0 over the training run
  * new sequences enter at the current maximum priority, so every episode is replayed at least once
    before its priority is ever reduced

Capacity is small enough (a few hundred episodes) that an O(N) sampling pass costs far less than one
forward pass of the network, so no sum-tree is needed; a flat array is used and is exact.

INTEGRITY: standard Schaul et al. (2016) proportional prioritization with the Kapturowski et al.
(2019) sequence-level unit. Nothing here is novel and nothing is claimed to be.
"""
import numpy as np


class SequencePER:
    """Proportional prioritized replay over whole episodes."""

    def __init__(self, capacity=500, alpha=0.6, beta0=0.4, beta1=1.0, beta_steps=1500, eps=1e-3,
                 rng=None):
        self.capacity = int(capacity)
        self.alpha, self.beta0, self.beta1 = float(alpha), float(beta0), float(beta1)
        self.beta_steps, self.eps = max(int(beta_steps), 1), float(eps)
        self.data = []                     # list of episode dicts
        self.prio = np.zeros(self.capacity, dtype=np.float64)
        self.pos = 0                       # next write position (circular)
        self._steps = 0                    # anneal counter
        self.rng = rng if rng is not None else np.random.default_rng(0)

    # ------------------------------------------------------------------ state
    def __len__(self):
        return len(self.data)

    @property
    def beta(self):
        f = min(self._steps / self.beta_steps, 1.0)
        return self.beta0 + f * (self.beta1 - self.beta0)

    @property
    def max_prio(self):
        n = len(self.data)
        return float(self.prio[:n].max()) if n else 1.0

    # ------------------------------------------------------------------ write
    def add(self, ep):
        """Insert one episode at the current maximum priority (so it is certain to be replayed)."""
        p = self.max_prio
        if len(self.data) < self.capacity:
            self.data.append(ep)
            self.prio[len(self.data) - 1] = p
            self.pos = len(self.data) % self.capacity
        else:
            self.data[self.pos] = ep
            self.prio[self.pos] = p
            self.pos = (self.pos + 1) % self.capacity

    # ------------------------------------------------------------------- read
    def sample(self, batch):
        """Return (episodes, indices, is_weights). is_weights are normalised to a maximum of 1."""
        n = len(self.data)
        assert n > 0, 'empty replay buffer'
        b = min(int(batch), n)
        pa = self.prio[:n] ** self.alpha
        s = pa.sum()
        probs = pa / s if s > 0 else np.full(n, 1.0 / n)
        idx = self.rng.choice(n, size=b, replace=False if b <= n else True, p=probs)
        w = (1.0 / (n * probs[idx])) ** self.beta
        w = w / w.max()
        return [self.data[i] for i in idx], idx, w.astype(np.float64)

    # ----------------------------------------------------------------- update
    def update(self, idx, td):
        """Set p_i = |delta_i| + eps for the sampled sequences."""
        td = np.asarray(td, dtype=np.float64)
        self.prio[np.asarray(idx, dtype=int)] = np.abs(td) + self.eps

    def anneal(self, k=1):
        self._steps += int(k)

    # ------------------------------------------------------------ diagnostics
    def stats(self):
        n = len(self.data)
        if not n:
            return dict(n=0, beta=self.beta, p_mean=float('nan'), p_max=float('nan'),
                        p_min=float('nan'))
        p = self.prio[:n]
        return dict(n=n, beta=round(self.beta, 3), p_mean=float(p.mean()),
                    p_max=float(p.max()), p_min=float(p.min()))


def _smoke():
    rng = np.random.default_rng(0)
    buf = SequencePER(capacity=10, beta_steps=100, rng=rng)
    for i in range(10):
        buf.add(dict(id=i))
    assert len(buf) == 10 and np.allclose(buf.prio[:10], 1.0), 'new entries must enter at max priority'

    # give entry 3 a huge TD error; it must dominate the sampling distribution
    buf.update([3], [100.0])
    hits = 0
    for _ in range(400):
        eps, idx, w = buf.sample(2)
        hits += int(3 in list(idx))
    assert hits > 300, f'high-priority sequence sampled only {hits}/400 times'

    # importance weights must offset the oversampling: the max weight is 1 and the
    # over-sampled entry must carry the SMALLEST weight
    eps, idx, w = buf.sample(10)
    assert abs(w.max() - 1.0) < 1e-12
    assert w[list(idx).index(3)] == w.min(), 'the oversampled sequence must get the smallest IS weight'

    # eviction is circular and capacity is respected
    for i in range(10, 25):
        buf.add(dict(id=i))
    assert len(buf) == 10 and {d['id'] for d in buf.data} == set(range(15, 25))

    # beta anneals 0.4 -> 1.0
    b0 = buf.beta
    buf.anneal(100)
    assert abs(b0 - 0.4) < 1e-12 and abs(buf.beta - 1.0) < 1e-12
    print(f'SMOKE OK — PER: max-priority insert, {hits}/400 priority hits, '
          f'IS weights normalised (min {w.min():.4f}, max {w.max():.4f}), '
          f'circular eviction, beta {b0:.2f}->{buf.beta:.2f}')


if __name__ == '__main__':
    _smoke()
