#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper2_controller.py — Paper-2 controller: branched QR-DQN + CVaR + optional LSTM, with
n-step prioritized sequence replay, trained on the training scenes and evaluated on held-out
scenes under common random numbers.

Architecture:
  obs(11) -> FC encoder -> {LSTM | feed-forward} -> per-branch DUELING QR-DQN heads emitting N
  quantiles per action, for the branched action {handover(2), power(3), spectrum(3)}.
Risk-sensitive control: action selection maximises CVaR_alpha of the return quantiles (the worst
alpha fraction of outcomes — the deep-blockage tail), NOT the mean. Loss = quantile-Huber (QR-DQN)
against an n-step Double-DQN target. This realises cliff -> tail -> CVaR: because deep blockage is
a tail event, optimise the tail.

R3 REVISION — four changes, all of which alter the reported numbers:

  C1  PRIORITIZED REPLAY IS NOW REAL.  The previous loop learned from `buf[-8:]`, the eight most
      recently collected episodes in collection order: a sliding on-policy window with no buffer,
      no priorities and no importance-sampling correction, while the manuscript claimed n-step
      prioritized experience replay.  Training now draws IS-weighted minibatches from
      paper2_per.SequencePER and writes back per-sequence TD errors as priorities.

  C2  TRUNCATION IS NO LONGER TREATED AS TERMINATION.  When the n-step window reached the end of
      an episode the target was set to the undiscounted partial return regardless of WHY the
      episode ended.  For episodes that ended at the time limit (the scene ran out of frames)
      that silently teaches the agent that the world stops — a value bias against surviving.  The
      target now bootstraps from the final state whenever the episode was truncated rather than
      terminated by sustained RLF.

  C3  EVALUATION IS OUT-OF-SAMPLE AND PAIRED.  Training uses `split='train'` (30 scenes) and every
      reported metric is measured on `split='test'` (15 scenes never seen in training), with
      reset seeds EVAL_SEED_BASE + i shared by every configuration, so all configurations are
      compared on byte-identical scenarios (common random numbers).  In-sample performance is
      also measured, so the generalisation gap can be reported rather than hidden.

  C4  RESOLUTION.  The RLF rate was estimated from 60 rollouts, i.e. it could only take multiples
      of 1.67 percentage points.  The default is now 500 rollouts.

Also added: per-decision inference latency measurement (single observation, batch 1, one CPU
thread), for comparison against the O-RAN Near-RT RIC 10 ms - 1 s control budget.

INTEGRITY: standard algorithms on the real-geometry environment; nothing fabricated.
"""
import os, sys, math, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from paper2_per import SequencePER

BRANCHES = [2, 3, 3]        # handover, power, spectrum
N_QUANT = 32
GAMMA = 0.99
NSTEP = 3
CVAR_ALPHA = 0.25          # optimise the worst 25% (the deep-blockage tail)
EVAL_SEED_BASE = 900_000_000   # fixed forever: every configuration evaluates on the same scenarios


class BranchedQRDQN(nn.Module):
    def __init__(self, obs_dim, branches=BRANCHES, n_quant=N_QUANT, hidden=128, lstm_hidden=128, use_lstm=True):
        # obs_dim is deliberately required: it is read from env.observation_space at every call
        # site, so the network can never silently disagree with the environment it trains on.
        super().__init__()
        self.branches, self.n_quant, self.use_lstm = branches, n_quant, use_lstm
        self.enc = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        if use_lstm:
            self.lstm = nn.LSTM(hidden, lstm_hidden, batch_first=True)
        else:                                          # memory ablation: feed-forward, no recurrence
            self.ff = nn.Sequential(nn.Linear(hidden, lstm_hidden), nn.ReLU())
        self.val = nn.ModuleList([nn.Linear(lstm_hidden, n_quant) for _ in branches])
        self.adv = nn.ModuleList([nn.Linear(lstm_hidden, nb * n_quant) for nb in branches])

    def forward(self, x, hidden=None):
        """x: (B,T,obs). Returns list of per-branch quantiles (B,T,n_act_b,n_quant) and LSTM hidden."""
        h = self.enc(x)
        if self.use_lstm:
            out, hidden = self.lstm(h, hidden)
        else:
            out, hidden = self.ff(h), None
        B, T, _ = out.shape
        qs = []
        for i, nb in enumerate(self.branches):
            v = self.val[i](out).view(B, T, 1, self.n_quant)
            a = self.adv[i](out).view(B, T, nb, self.n_quant)
            qs.append(v + a - a.mean(dim=2, keepdim=True))
        return qs, hidden


def cvar_scores(q_branch, alpha):
    """q_branch: (..., n_act, n_quant) -> CVaR_alpha per action (mean of lowest alpha-fraction quantiles)."""
    k = max(1, int(round(alpha * q_branch.shape[-1])))
    low, _ = torch.sort(q_branch, dim=-1)
    return low[..., :k].mean(dim=-1)


def greedy_actions(qs, alpha):
    """Per-branch CVaR-greedy actions from quantiles qs[i]: (B,T,n_act,n_quant)."""
    return [cvar_scores(q, alpha).argmax(dim=-1) for q in qs]


def quantile_huber(pred, target, taus, kappa=1.0):
    """pred,target: (B,n_quant). Quantile-Huber loss (QR-DQN)."""
    u = target.unsqueeze(1) - pred.unsqueeze(2)                     # (B, Npred, Ntar)
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    rho = (taus.view(1, -1, 1) - (u.detach() < 0).float()).abs() * huber / kappa
    return rho.sum(dim=1).mean(dim=1).mean()


class Agent:
    def __init__(self, obs_dim, alpha=CVAR_ALPHA, lr=5e-4, device='cpu', use_lstm=True):
        self.net = BranchedQRDQN(obs_dim, use_lstm=use_lstm).to(device)
        self.tgt = BranchedQRDQN(obs_dim, use_lstm=use_lstm).to(device); self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.taus = torch.tensor([(i + 0.5) / N_QUANT for i in range(N_QUANT)], device=device)
        self.alpha, self.device = alpha, device

    @torch.no_grad()
    def act(self, obs, hidden, eps):
        x = torch.tensor(obs, dtype=torch.float32, device=self.device).view(1, 1, -1)
        qs, hidden = self.net(x, hidden)
        if eps > 0.0 and np.random.random() < eps:
            a = [np.random.randint(nb) for nb in BRANCHES]
        else:
            a = [int(g[0, 0]) for g in greedy_actions(qs, self.alpha)]
        return a, hidden

    # ------------------------------------------------------------------ learning
    def _seq_loss(self, ep):
        """Quantile-Huber loss and mean |n-step TD error| for one episode (sequence).

        ep: dict(obs:(T+1,obs), acts:(T,3), rews:(T,), term:bool).
        `term` is True only when the episode ended in sustained RLF (a true absorbing state).  When
        it is False the episode ended at the scene time limit, so the n-step target must bootstrap
        from the final state instead of pretending the return stopped there (fix C2).
        """
        obs = torch.tensor(ep['obs'], dtype=torch.float32, device=self.device).unsqueeze(0)   # (1,T+1,obs)
        acts = torch.tensor(ep['acts'], dtype=torch.long, device=self.device)                 # (T,3)
        rews = torch.tensor(ep['rews'], dtype=torch.float32, device=self.device)              # (T,)
        term = bool(ep.get('term', True))
        T = acts.shape[0]
        qs_all, _ = self.net(obs)                                    # per-branch (1,T+1,n_act,nq)
        with torch.no_grad():
            qs_tgt, _ = self.tgt(obs)
            qs_on, _ = self.net(obs)
            astar = greedy_actions(qs_on, self.alpha)                # Double-DQN: online picks a*
        loss, td = 0.0, 0.0
        for i, nb in enumerate(BRANCHES):
            pred = qs_all[i][0, torch.arange(T), acts[:, i]]         # (T, nq) chosen-action quantiles at s_t
            tgt = torch.zeros(T, N_QUANT, device=self.device)
            for t in range(T):
                R = 0.0; g = 1.0; k = 0
                while k < NSTEP and t + k < T:
                    R += g * rews[t + k].item(); g *= GAMMA; k += 1
                if t + k < T:                                        # bootstrap from an interior state
                    tgt[t] = R + g * qs_tgt[i][0, t + k, astar[i][0, t + k]]
                elif not term:                                       # C2: time-limit truncation
                    tgt[t] = R + g * qs_tgt[i][0, T, astar[i][0, T]]
                else:                                                # true terminal (sustained RLF)
                    tgt[t] = R
            tgt = tgt.detach()
            loss = loss + quantile_huber(pred, tgt, self.taus)
            # priority signal: distributional TD error reduced to a scalar via the quantile means
            td = td + float((tgt.mean(dim=1) - pred.detach().mean(dim=1)).abs().mean())
        return loss, td / len(BRANCHES)

    def learn(self, episodes, is_w=None):
        """One optimizer step on an importance-weighted minibatch of sequences.

        Returns (weighted mean loss, per-sequence |TD error|) so the caller can write the TD errors
        back to the replay buffer as priorities.
        """
        self.net.train()
        if is_w is None:
            is_w = np.ones(len(episodes), dtype=np.float64)
        is_w = np.asarray(is_w, dtype=np.float64)
        wsum = float(is_w.sum()) or 1.0
        self.opt.zero_grad()
        total, tds = 0.0, []
        for ep, w in zip(episodes, is_w):
            l, td = self._seq_loss(ep)
            (float(w) / wsum * l).backward()          # accumulate; graph freed per sequence
            total += float(l.detach()) * float(w) / wsum
            tds.append(td)
        nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
        self.opt.step()
        return total, np.array(tds, dtype=np.float64)

    def sync(self):
        self.tgt.load_state_dict(self.net.state_dict())


def rollout(env, agent, eps, learn_buf=None, seed=None, latency=None, trace=None):
    """One episode. `seed` makes the scene and the whole fading realisation deterministic (CRN).

    `latency`, if a list, receives the wall-clock cost in milliseconds of every act() call.
    `trace`, if a list, receives one per-episode record of the actuator state (mean/final
    transmit power, mean sub-band, handover count). Added 28 Aug 2026 because the
    convergence basin is DEFINED by mean transmit power and it was never recorded per
    episode -- only per-episode return was, and return carries the basin only weakly
    (within-config AUC 0.714 at episode 500 against a 0.80 bar).
    Both are pure observers: default None reproduces the previous behaviour exactly,
    and neither consumes randomness, so seeds are unaffected.
    Returns (episode return, terminated) where terminated == sustained RLF.
    """
    obs, info = env.reset(seed=seed) if seed is not None else env.reset()
    hidden = None
    O, Ac, Re = [obs], [], []
    pts, rbs, hos = [], [], 0
    done = trunc = False
    while not (done or trunc):
        if latency is None:
            a, hidden = agent.act(obs, hidden, eps)
        else:
            t0 = time.perf_counter()
            a, hidden = agent.act(obs, hidden, eps)
            latency.append((time.perf_counter() - t0) * 1000.0)
        obs, r, done, trunc, info = env.step(a)
        if trace is not None:
            pts.append(float(env.pt)); rbs.append(float(env.rb)); hos += int(a[0] == 1)
        O.append(obs); Ac.append(a); Re.append(r)
    ep = dict(obs=np.array(O, dtype=np.float32), acts=np.array(Ac), rews=np.array(Re, dtype=np.float32),
              term=bool(done))
    if learn_buf is not None:
        (learn_buf.add if hasattr(learn_buf, 'add') else learn_buf.append)(ep)
    if trace is not None:
        trace.append(dict(pt_mean=float(np.mean(pts)) if pts else float('nan'),
                          pt_last=float(pts[-1]) if pts else float('nan'),
                          rb_mean=float(np.mean(rbs)) if rbs else float('nan'),
                          ho=int(hos), term=bool(done), steps=len(Re)))
    return sum(Re), bool(done)                # done == terminated == sustained RLF this episode


# ---------------------------------------------------------------------- evaluation
def evaluate(env, agent, n, seed_base=EVAL_SEED_BASE, latency=None):
    """Greedy evaluation over n episodes with common random numbers.

    Episode i always uses reset seed `seed_base + i`, so every configuration and every training
    seed is scored on exactly the same n scenarios with the same fading realisations. Returns a
    per-episode DataFrame — the paired statistics are computed from it downstream.
    """
    import pandas as pd
    rows = []
    for i in range(n):
        obs, info = env.reset(seed=seed_base + i)
        hidden, R, steps = None, 0.0, 0
        pts, rbs, nho = [], [], 0
        done = trunc = False
        while not (done or trunc):
            if latency is None:
                a, hidden = agent.act(obs, hidden, 0.0)
            else:
                t0 = time.perf_counter()
                a, hidden = agent.act(obs, hidden, 0.0)
                latency.append((time.perf_counter() - t0) * 1000.0)
            obs, r, done, trunc, i2 = env.step(a)
            R += r; steps += 1
            pts.append(i2['pt']); rbs.append(i2['rb']); nho += int(i2['ho'])
        rows.append(dict(idx=i, eval_seed=seed_base + i, scene=info['scene'],
                         ret=round(float(R), 6), rlf=int(bool(done)), steps=steps,
                         mean_pt=round(float(np.mean(pts)), 4), mean_rb=round(float(np.mean(rbs)), 4),
                         n_ho=nho))
    return pd.DataFrame(rows)


def summarize(df, tag=''):
    r = df['ret'].to_numpy(dtype=float)
    k = max(1, int(round(0.25 * len(r))))
    p = (tag + '_') if tag else ''
    out = {p + 'return_mean': round(float(r.mean()), 4),
           p + 'return_std': round(float(r.std(ddof=1)), 4) if len(r) > 1 else 0.0,
           p + 'cvar25': round(float(np.sort(r)[:k].mean()), 4),
           p + 'rlf_rate': round(float(100.0 * df['rlf'].mean()), 4),
           p + 'n_eval': int(len(df))}
    # resource efficiency, measured rather than asserted: what the policy actually spends
    for c, nm in (('mean_pt', 'pt_dbm'), ('mean_rb', 'rb'), ('n_ho', 'ho_per_ep'), ('steps', 'steps')):
        if c in df.columns:
            out[p + nm] = round(float(df[c].mean()), 4)
    return out


def latency_stats(lat):
    a = np.asarray(lat, dtype=float)
    if a.size == 0:
        return {}
    return {'lat_mean_ms': round(float(a.mean()), 4), 'lat_p50_ms': round(float(np.percentile(a, 50)), 4),
            'lat_p95_ms': round(float(np.percentile(a, 95)), 4), 'lat_p99_ms': round(float(np.percentile(a, 99)), 4),
            'lat_max_ms': round(float(a.max()), 4), 'lat_n': int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', default='dair_geometry_pool.csv')
    ap.add_argument('--p1', default=None)
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--alpha', type=float, default=CVAR_ALPHA,
                    help='CVaR level; RISK ablation: --alpha 1.0 = expected-value (mean) Dn-DQN')
    ap.add_argument('--ff', action='store_true', help='MEMORY ablation: feed-forward (no LSTM)')
    ap.add_argument('--no_ant', action='store_true', help='ANTICIPATION ablation: zero the N1 features')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=None, help='save trained model (.pt) + return history + per-episode eval + metrics')
    ap.add_argument('--eval_n', type=int, default=500, help='held-out evaluation rollouts (common random numbers)')
    ap.add_argument('--eval_n_train', type=int, default=200, help='in-sample rollouts, for the generalisation gap')
    ap.add_argument('--seed', type=int, default=0, help='random seed (multi-seed runs)')
    # --- prioritized replay
    ap.add_argument('--per_capacity', type=int, default=500, help='replay capacity in episodes')
    ap.add_argument('--per_alpha', type=float, default=0.6)
    ap.add_argument('--per_beta0', type=float, default=0.4)
    ap.add_argument('--batch', type=int, default=8, help='sequences per gradient step')
    ap.add_argument('--warmup', type=int, default=32, help='episodes collected before learning starts')
    ap.add_argument('--updates_per_ep', type=int, default=2, help='gradient steps per collected episode')
    ap.add_argument('--target_sync', type=int, default=0, help='target-net sync period in episodes (0 = episodes//6)')
    ap.add_argument('--no_per', action='store_true', help='replay ablation: uniform sampling, no IS weights')
    ap.add_argument('--threads', type=int, default=1, help='torch CPU threads (1 = deployment condition)')
    ap.add_argument('--w_pwr', type=float, default=None,
                    help='override the reward transmit-power penalty weight (paper2_env.W_PWR, default 0.15). '
                         'Reward-weight sensitivity sweep, 30 Aug 2026.')
    a = ap.parse_args()
    torch.set_num_threads(max(1, a.threads))
    if a.p1:
        sys.path.insert(0, a.p1)
    # set BEFORE the env is imported: paper2_env reads P2_W_PWR at module scope
    if a.w_pwr is not None:
        os.environ['P2_W_PWR'] = repr(float(a.w_pwr))
    import pandas as pd, paper2_env as PE
    # G1: fail in the first second, not after six minutes of training on the wrong reward
    if a.w_pwr is not None and abs(PE.W_PWR - a.w_pwr) > 1e-12:
        sys.exit(f'ABORT: asked for W_PWR={a.w_pwr!r} but paper2_env.W_PWR is {PE.W_PWR!r}. '
                 'The override did not take -- refusing to train.')
    pool = pd.read_csv(a.pool)
    # C3: train on the training scenes only; every reported number comes from the held-out scenes
    env = PE.MultiRSUBlockageEnv(pool, seed=a.seed, mask_anticipation=a.no_ant, split='train')
    env_te = PE.MultiRSUBlockageEnv(pool, seed=a.seed + 10_000, mask_anticipation=a.no_ant, split='test')
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    agent = Agent(obs_dim=env.observation_space.shape[0], alpha=a.alpha, use_lstm=not a.ff)
    buf = SequencePER(capacity=a.per_capacity, alpha=(0.0 if a.no_per else a.per_alpha),
                      beta0=(1.0 if a.no_per else a.per_beta0), beta_steps=max(a.episodes, 1),
                      rng=np.random.default_rng(a.seed + 777))
    sync_every = a.target_sync if a.target_sync > 0 else max(a.episodes // 6, 1)
    print(f"config: {'FF' if a.ff else 'LSTM'} | {'mean' if a.alpha >= 0.999 else f'CVaR{a.alpha}'} | "
          f"anticipation={'OFF' if a.no_ant else 'ON'} | replay={'uniform' if a.no_per else 'PER'} "
          f"(cap {a.per_capacity}, batch {a.batch}, {a.updates_per_ep} upd/ep, warmup {a.warmup})")
    print(f"reward weights: W_REL={PE.W_REL} RLF_PEN={PE.RLF_PEN} W_PWR={PE.W_PWR} "
          f"W_SPEC={PE.W_SPEC} W_HO={PE.W_HO}" + ("   <-- W_PWR OVERRIDDEN" if a.w_pwr is not None else ""))
    print(f"train scenes={len(env.scene_ids)} held-out scenes={len(env_te.scene_ids)} | "
          f"quant={N_QUANT} nstep={NSTEP} gamma={GAMMA} | target sync every {sync_every} ep", flush=True)

    hist, loss, t0 = [], float('nan'), time.time()
    ptrace = []                       # per-episode actuator trace (see rollout docstring)
    for ep in range(a.episodes):
        eps = max(0.05, 1.0 - ep / max(a.episodes * 0.7, 1))
        ret, _ = rollout(env, agent, eps, learn_buf=buf, trace=ptrace)
        hist.append(ret)
        if len(buf) >= a.warmup:
            for _ in range(a.updates_per_ep):
                eps_b, idx, w = buf.sample(a.batch)
                loss, td = agent.learn(eps_b, w)
                buf.update(idx, td)
            buf.anneal()
        if (ep + 1) % sync_every == 0:
            agent.sync()
            st = buf.stats()
            print(f"  ep {ep+1:5d} | eps {eps:.2f} | loss {loss:8.3f} | recent return "
                  f"{np.mean(hist[-25:]):7.2f} | buf {st['n']:4d} beta {st['beta']:.2f} "
                  f"p[{st['p_min']:.3f},{st['p_max']:.3f}] | {time.time()-t0:6.0f}s", flush=True)

    lat = []
    ev_te = evaluate(env_te, agent, a.eval_n, latency=lat)
    ev_tr = evaluate(env, agent, a.eval_n_train)
    m = dict(seed=a.seed, alpha=a.alpha, lstm=int(not a.ff), anticipation=int(not a.no_ant),
             per=int(not a.no_per), episodes=a.episodes, w_pwr=float(PE.W_PWR),
             train_s=round(time.time() - t0, 1))
    m.update(summarize(ev_te))                       # held-out: the reported numbers
    m.update(summarize(ev_tr, 'insample'))
    m['gap_return'] = round(m['insample_return_mean'] - m['return_mean'], 4)
    m['gap_rlf'] = round(m['rlf_rate'] - m['insample_rlf_rate'], 4)
    m.update(latency_stats(lat))
    print(f"HELD-OUT  return={m['return_mean']:.2f}  CVaR25={m['cvar25']:.2f}  RLF={m['rlf_rate']:.2f}%  "
          f"(n={m['n_eval']})")
    print(f"IN-SAMPLE return={m['insample_return_mean']:.2f}  CVaR25={m['insample_cvar25']:.2f}  "
          f"RLF={m['insample_rlf_rate']:.2f}%  (n={m['insample_n_eval']}) | gap {m['gap_return']:+.2f} return, "
          f"{m['gap_rlf']:+.2f} pp RLF")
    print(f"latency/decision: mean {m.get('lat_mean_ms', float('nan')):.3f} ms  "
          f"p95 {m.get('lat_p95_ms', float('nan')):.3f} ms  p99 {m.get('lat_p99_ms', float('nan')):.3f} ms")
    if a.out:
        torch.save(agent.net.state_dict(), a.out + '.pt')
        _rt = pd.DataFrame({'episode': range(len(hist)), 'return': hist})
        if len(ptrace) == len(hist):      # pt_mean is the quantity that DEFINES the basin
            _rt = pd.concat([_rt, pd.DataFrame(ptrace)], axis=1)
        _rt.to_csv(a.out + '_returns.csv', index=False)
        ev_te.to_csv(a.out + '_eval.csv', index=False)
        ev_tr.to_csv(a.out + '_eval_train.csv', index=False)
        pd.DataFrame([m]).to_csv(a.out + '_metrics.csv', index=False)
        print(f"saved -> {a.out}.pt / _returns.csv / _eval.csv / _eval_train.csv / _metrics.csv")
    if a.smoke:
        print('SMOKE OK — QR-DQN + CVaR + prioritized sequence replay trains end-to-end, '
              'evaluated out-of-sample under common random numbers.')


if __name__ == '__main__':
    main()
