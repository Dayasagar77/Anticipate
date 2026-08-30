"""
env_6g_v6.py — v6 = v5 with a PHYSICALLY-CORRECT (sustained) radio-link-failure model.

Motivation
----------
v5 declared RLF and permanently terminated the episode the instant a SINGLE 40 ms
sample of the (fast-faded) serving RSRP fell below -92 dBm. Because the alpha-mu
small-scale fading is drawn independently every 40 ms step, over a ~67-step episode
the probability that at least one sample dips below the floor is ~80% — so ~80% of
episodes "failed", even at maximum transmit power. That is a single-sample artifact,
not link failure: 3GPP TS 38.133 / 38.331 declare RLF only after N310 CONSECUTIVE
out-of-sync indications (then a T310 timer), and RSRP is itself an L3-filtered
average that does not track a single instantaneous fade.

v6 changes
----------
1. Sustained RLF: an episode terminates on radio-link failure only after N310
   CONSECUTIVE sub-threshold samples (out-of-sync run). A single/short dip that
   recovers does not end the episode (the counter resets on any in-sync sample).
2. Reward: the v5 single-sample -1000 "cliff" for RSRP < -92 is replaced by the
   SAME graded urgency penalty used just above the floor, so the agent is still
   discouraged from lingering in poor coverage but is not hit by an artifactual
   cliff on a transient fade. The terminal RLF penalty is applied by the training
   wrapper only on a *sustained* failure (see train_phase2_v6.Phase2Env_v6).

Everything else (channel model, geometry, actions, observation) is identical to v5.
N310 = 3 (=120 ms of sustained out-of-sync) is a conservative, 3GPP-style choice;
real T310 windows are longer, so this OVER-counts RLF if anything.
"""
import numpy as np
from env_6g_v5 import (
    SubTHzEnv6G,
    A3_THRESH_DBM, HO_PROACTIVE_THRESH, RLF_THRESH_DBM,
    URGE_BASE, URGE_DELTA, MAX_STEPS,
    PT_MIN_DBM, PT_MAX_DBM, RB_MIN, RB_MAX,
    HANDOVER,
)

# 3GPP-style sustained out-of-sync count to declare RLF (N310-like).
N310_DEFAULT = 3


class SubTHzEnv6G_v6(SubTHzEnv6G):
    """v5 environment with sustained (3GPP-style) RLF instead of single-sample."""

    def __init__(self, vehicle_trajectory, n310=N310_DEFAULT, **kw):
        self.n310 = int(n310)
        super().__init__(vehicle_trajectory, **kw)

    def _reset_state(self):
        super()._reset_state()
        self._consec_below = 0

    def _compute_reward(self, rsrp_dbm, action_taken):
        """Identical to v5 EXCEPT the single-sample RSRP<-92 '-1000' cliff is
        replaced by the graded urgency penalty (continuous through the floor)."""
        r_pwr_raw  = (PT_MAX_DBM - self.pt_dbm)  / (PT_MAX_DBM - PT_MIN_DBM) * 10.0
        r_spec_raw = (RB_MAX     - self.rb_count) / (RB_MAX     - RB_MIN)     * 10.0

        if action_taken == HANDOVER:
            r_mob = +10.0 if (HO_PROACTIVE_THRESH >= rsrp_dbm >= RLF_THRESH_DBM) else -5.0
        else:
            if rsrp_dbm > A3_THRESH_DBM:
                r_mob = +1.0
            else:
                # graded penalty for ALL rsrp <= A3, continuous through -92
                depth = abs(rsrp_dbm - A3_THRESH_DBM)
                r_mob = -(URGE_BASE + URGE_DELTA * depth)

        self._last_r_mobility = r_mob
        self._last_r_power    = r_pwr_raw
        self._last_r_spectrum = r_spec_raw
        return self.w_mob * r_mob + self.w_pwr * r_pwr_raw + self.w_spec * r_spec_raw

    def step(self, action):
        obs, reward, _term_v5, truncated, info = super().step(action)

        # sustained out-of-sync tracking
        if info['rsrp_dbm'] < RLF_THRESH_DBM:
            self._consec_below += 1
        else:
            self._consec_below = 0
        sustained_rlf = self._consec_below >= self.n310

        terminated = (
            sustained_rlf or
            self.step_idx >= self.T or
            self.episode_step >= MAX_STEPS
        )
        info['sustained_rlf'] = bool(sustained_rlf)
        info['consec_below']  = int(self._consec_below)
        return obs, reward, terminated, truncated, info
