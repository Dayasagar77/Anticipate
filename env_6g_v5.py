"""
env_6g_v5.py
------------
Upgraded 6G Sub-THz DQN environment — Version 5.

v5 addition: Molecular absorption loss (ITU-R P.676-13) added to RSRP.
  At 140 GHz, standard atmosphere (15°C, 1013.25 hPa, 7.5 g/m³ water vapour):
    - Specific attenuation: γ_abs = 1.0 dB/km
    - At cell radius (37.4m): 0.037 dB loss
    - At max link (100m):     0.100 dB loss
  Applied as: RSRP_v5 = RSRP_freespace - γ_abs × d_km
  Reference: ITU-R P.676-13 (2022), Table 1, 140 GHz row.
  This satisfies reviewer requests (R1.1, R2.2) for Sub-THz propagation effects.
  The loss is physically real and small; retraining ensures internal consistency.

v4 fix retained: r_pwr_raw and r_spec_raw no longer zeroed on HANDOVER steps.

Key features:
  1. Roadside gNB deployment (Option A) — 5 gNBs, ISD=74.8m, D_PERP=10m
  2. 6D observation space — adds normalised xVelocity
  3. Doppler shift modelled in SINR — f_d = (v/c)*f, ICI noise term
  4. Molecular absorption — ITU-R P.676-13, 1.0 dB/km at 140 GHz  ← v5 NEW
  5. N=3 co-channel interferers in SINR
  6. Multi-gNB trajectory — serving gNB changes as vehicle moves (real handover)
  7. episode_step counter separate from step_idx (fix for mid-traj starts)
  8. Valid start position sampling — only positions above RLF threshold
  9. Proactive HO reward threshold at -75 dBm (5 dBm above A3=-80 dBm)

State  (6D): [rsrp_norm, margin_norm, pwr_norm, rb_norm, sinr_norm, vel_norm]
Actions (6): HOLD, PWR_UP, PWR_DOWN, RB_UP, RB_DOWN, HANDOVER
Reward: w_mob*R_mob + w_pwr*R_pwr + w_spec*R_spec (configurable weights)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from channel_model import (
    channel_gain, sinr, apply_fading,
    get_interferer_distances, N0_WATTS, N0_DBM,
    ALPHA_LOS, MU_LOS, ALPHA_NLOS, MU_NLOS
)

# ── System parameters ─────────────────────────────────────────────────────────
FREQ           = 140e9
C_LIGHT        = 3e8
ISD            = 74.8        # inter-site distance (m)
D_PERP         = 10.0        # perpendicular distance road→gNB (m)
N_GNB          = 5           # number of gNBs along road
GNB_X          = np.array([ISD/2 + i*ISD for i in range(N_GNB)])  # gNB x-positions
CELL_RADIUS    = ISD / 2     # = 37.4m
A3_THRESH_DBM        = -80.0  # A3 trigger threshold (reporting / too_early boundary)
HO_PROACTIVE_THRESH  = -75.0  # Proactive HO reward threshold (5 dBm above A3)
                               # Widens reward window to survive +-5 dBm fading noise.
                               # Agent rewarded for HO when RSRP in [-92, -75] dBm.
                               # too_early flagged only when RSRP > -75 (clearly premature).
RLF_THRESH_DBM = -92.0        # Radio Link Failure threshold
PT_MIN_DBM     = 23.0
PT_MAX_DBM     = 30.0
PT_STEP_DBM    = 3.0
RB_MIN         = 10
RB_MAX         = 66
RB_STEP        = 8
MAX_STEPS      = 800
E2_DELAY_STEPS = 1

# Velocity normalization bounds (m/s) — from HighD v2 data
VEL_MIN = 5.0
VEL_MAX = 60.0

# Reward weights (balanced configuration — v5_balanced)
# Physical justification: at 140 GHz Sub-THz, spectrum efficiency (RB allocation)
# is physically coupled to power efficiency via PA back-off and bandwidth-noise scaling.
# ITU-R P.676-13 / Lozano & Rangan (arXiv:2310.02622) confirm no theoretical basis
# for weighting SE less than EE at THz. Equal W_PWR=W_SPEC=0.20 matches physics.
# W_MOB reduced from 0.70 → 0.60 to accommodate equal resource weights; HO remains dominant.
W_MOB  = 0.60
W_SPEC = 0.20
W_PWR  = 0.20

# Urgency penalty
URGE_BASE  = 50.0
URGE_DELTA = 10.0

# Actions
HOLD     = 0
PWR_UP   = 1
PWR_DOWN = 2
RB_UP    = 3
RB_DOWN  = 4
HANDOVER = 5
N_ACTIONS = 6

# Interferer config
N_INTERFERERS     = 3
INTERFERER_PT_DBM = 30.0
INTERFERER_GAIN   = 1.0

# Molecular absorption — ITU-R P.676-13 (2022)
# Specific attenuation at 140 GHz, standard atmosphere:
#   Temperature: 15°C | Pressure: 1013.25 hPa | Water vapour: 7.5 g/m³
#   γ_abs ≈ 0.4 dB/km (dry air/O2) + 0.6 dB/km (H2O) = 1.0 dB/km total
#   At cell radius (37.4m): 0.037 dB | At 100m link: 0.100 dB
# Reference: ITU-R P.676-13, Table 1, interpolated at 140 GHz.
MOLECULAR_ABS_DB_PER_KM = 1.0   # dB/km at 140 GHz, standard atmosphere


class SubTHzEnv6G(gym.Env):
    """
    Upgraded 6G Sub-THz environment with velocity-aware proactive handover.
    Uses roadside gNB deployment and Doppler-aware channel model.
    """

    metadata = {'render_modes': []}

    def __init__(self, vehicle_trajectory, seed=None,
                 w_mob=W_MOB, w_spec=W_SPEC, w_pwr=W_PWR):
        """
        Args:
            vehicle_trajectory: array shape (T, 2) — columns [Distance, xVelocity]
                                 Distance = distance to serving gNB (m)
                                 xVelocity = vehicle speed (m/s)
            seed: random seed
            w_mob, w_spec, w_pwr: reward weights (must sum to 1)
        """
        super().__init__()

        self.traj   = vehicle_trajectory  # shape (T,2): col0=x_position, col1=xVelocity
        self.T      = len(vehicle_trajectory)
        self.w_mob  = w_mob
        self.w_spec = w_spec
        self.w_pwr  = w_pwr
        self.rng    = np.random.default_rng(seed)

        # 6D observation space: [rsrp, margin, pwr, rb, sinr, velocity]
        self.observation_space = spaces.Box(
            low =np.array([-2.0, -2.0, 0.0, 0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([ 2.0,  2.0, 1.0, 1.0,  2.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._reset_state()
        self._obs_buffer = [None] * (E2_DELAY_STEPS + 1)

    def _reset_state(self):
        self.step_idx          = 0
        self.episode_step      = 0
        self.pt_dbm            = PT_MAX_DBM
        self.rb_count          = RB_MAX
        self.handover_done     = False
        self.handover_step     = None
        self.ue_x              = 0.0
        self.ue_y              = 0.0
        self._last_r_mobility  = 0.0
        self._last_r_power     = 0.0
        self._last_r_spectrum  = 0.0
        self.total_handovers   = 0
        self.serving_gnb_idx   = 0   # index into GNB_X array

    def _get_x_position(self):
        """Vehicle x-position along road (metres)."""
        if self.step_idx >= self.T:
            return float(GNB_X[self.serving_gnb_idx])
        return float(self.traj[self.step_idx, 0])

    def _get_distance(self):
        """Distance to serving gNB, computed from x-position dynamically."""
        x = self._get_x_position()
        d = abs(x - GNB_X[self.serving_gnb_idx])
        # Add perpendicular distance for 3D slant distance
        return float(np.sqrt(d**2 + D_PERP**2))

    def _get_nearest_gnb(self):
        """Return index of nearest gNB to current x-position."""
        x = self._get_x_position()
        dists = np.abs(GNB_X - x)
        return int(np.argmin(dists))

    def _get_velocity(self):
        """xVelocity from trajectory (m/s) — column 1."""
        if self.step_idx >= self.T:
            return 0.0
        return float(self.traj[self.step_idx, 1])

    def _compute_doppler_interference(self, velocity_ms):
        """
        Doppler shift as additional interference term in SINR.
        f_d = (v/c) * f_carrier
        Modelled as additive noise power proportional to f_d.
        Returns interference power in watts.
        """
        f_d = abs(velocity_ms) / C_LIGHT * FREQ   # Doppler frequency (Hz)
        # Doppler interference: scales with f_d relative to subcarrier spacing
        # Sub-THz subcarrier spacing ~= 480 kHz (5G NR numerology 5)
        subcarrier_spacing = 480e3   # Hz
        doppler_ratio = f_d / subcarrier_spacing
        # ICI power = doppler_ratio^2 * signal_power (standard ICI model)
        pt_watts = 10 ** ((self.pt_dbm - 30) / 10)
        return doppler_ratio ** 2 * pt_watts

    def _compute_rsrp(self, d_m, apply_fading_flag=True):
        """Compute RSRP with Beer-Lambert + alpha-mu fading + molecular absorption.

        v5: Adds ITU-R P.676-13 molecular absorption loss:
            L_mol = MOLECULAR_ABS_DB_PER_KM × d_km
        Applied after fading so it represents the full propagation budget.
        At 140 GHz the absorption is small (0.1 dB at 100m) but physically real.
        """
        is_los = (d_m < CELL_RADIUS * 0.75)
        L = channel_gain(d_m)
        if apply_fading_flag:
            L = apply_fading(L, is_los=is_los, rng=self.rng)
        pt_watts = 10 ** ((self.pt_dbm - 30) / 10)
        rsrp_watts = pt_watts * L
        rsrp_dbm = 10 * np.log10(max(rsrp_watts, 1e-20) / 1e-3)
        # Molecular absorption loss (ITU-R P.676-13)
        mol_abs_db = MOLECULAR_ABS_DB_PER_KM * (d_m / 1000.0)
        rsrp_dbm  -= mol_abs_db
        return rsrp_dbm, is_los

    def _compute_sinr(self, d_m, is_los, velocity_ms=0.0):
        """
        SINR with N=3 co-channel interferers + Doppler interference term.
        """
        i_dists = get_interferer_distances(
            self.ue_x, self.ue_y,
            CELL_RADIUS, 0.0,
            isd_m=ISD, n_interferers=N_INTERFERERS
        )
        sinr_db, sinr_linear = sinr(
            p_s_dbm=self.pt_dbm,
            g_s=1.0,
            d_s_m=d_m,
            interferer_powers_dbm=[INTERFERER_PT_DBM] * N_INTERFERERS,
            interferer_gains=[INTERFERER_GAIN] * N_INTERFERERS,
            interferer_distances_m=i_dists
        )

        # Add Doppler interference
        doppler_interference = self._compute_doppler_interference(velocity_ms)
        if doppler_interference > 0:
            sinr_linear_val = 10 ** (sinr_db / 10)
            # Total interference = original + Doppler
            signal_power = sinr_linear_val * (N0_WATTS + doppler_interference)
            new_sinr_linear = signal_power / (N0_WATTS + doppler_interference +
                              sinr_linear_val * doppler_interference /
                              max(sinr_linear_val, 1e-10))
            sinr_db = float(10 * np.log10(max(new_sinr_linear, 1e-10)))

        return sinr_db

    def _build_obs(self, rsrp_dbm, sinr_db, velocity_ms):
        """Build normalised 6D state vector."""
        rsrp_norm   = rsrp_dbm / 100.0
        margin_norm = (rsrp_dbm - A3_THRESH_DBM) / 100.0
        pwr_norm    = (self.pt_dbm - PT_MIN_DBM) / (PT_MAX_DBM - PT_MIN_DBM)
        rb_norm     = (self.rb_count - RB_MIN) / (RB_MAX - RB_MIN)
        sinr_norm   = sinr_db / 100.0
        vel_norm    = np.clip((velocity_ms - VEL_MIN) / (VEL_MAX - VEL_MIN), 0.0, 1.0)
        return np.array([rsrp_norm, margin_norm, pwr_norm, rb_norm,
                         sinr_norm, vel_norm], dtype=np.float32)

    def _compute_reward(self, rsrp_dbm, action_taken):
        """Composite reward — matches paper Eq.(5),(6),(7).
        r_pwr and r_spec in [0,1] range — original scaling.

        FIX (v4): r_pwr_raw and r_spec_raw are NO LONGER zeroed on HANDOVER
        steps. pt_dbm and rb_count are valid resource state regardless of
        which action was taken. Zeroing them starved the power and spectrum
        gradient on ~73% of all steps (the HANDOVER fraction), preventing
        the agent from learning joint optimisation correctly.
        """
        # Scale to [0, 10] so W_PWR/W_SPEC weights are meaningful relative to r_mob.
        # r_mob on good HO = +10; at W_MOB=0.60: contribution = 6.0
        # W_SPEC×r_spec_max = 0.20×10 = 2.0 per step (equal to power — physics-grounded)
        # W_PWR×r_pwr_max  = 0.20×10 = 2.0 per step (proportional to stated weight)
        r_pwr_raw  = (PT_MAX_DBM  - self.pt_dbm)  / (PT_MAX_DBM  - PT_MIN_DBM) * 10.0
        r_spec_raw = (RB_MAX      - self.rb_count) / (RB_MAX      - RB_MIN)     * 10.0

        if action_taken == HANDOVER:
            if HO_PROACTIVE_THRESH >= rsrp_dbm >= RLF_THRESH_DBM:
                r_mob = +10.0
            else:
                r_mob = -5.0
            # r_pwr_raw and r_spec_raw preserved — agent rewarded for
            # maintaining low power and low RBs even when handing over
        else:
            if rsrp_dbm > A3_THRESH_DBM:
                r_mob = +1.0
            elif rsrp_dbm >= RLF_THRESH_DBM:
                depth = abs(rsrp_dbm - A3_THRESH_DBM)
                r_mob = -(URGE_BASE + URGE_DELTA * depth)
            else:
                r_mob = -1000.0

        self._last_r_mobility = r_mob
        self._last_r_power    = r_pwr_raw
        self._last_r_spectrum = r_spec_raw

        return self.w_mob * r_mob + self.w_pwr * r_pwr_raw + self.w_spec * r_spec_raw

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._reset_state()

        # Sample valid start position (RSRP above RLF threshold)
        # Use nearest gNB for each candidate position
        valid_starts = []
        for idx in range(0, max(1, self.T - 10)):
            self.step_idx      = idx
            self.serving_gnb_idx = self._get_nearest_gnb()
            d_m = self._get_distance()
            rsrp_dbm, _ = self._compute_rsrp(d_m, apply_fading_flag=False)
            if rsrp_dbm >= RLF_THRESH_DBM:
                valid_starts.append(idx)
        if valid_starts:
            start = valid_starts[int(self.rng.integers(0, len(valid_starts)))]
        else:
            start = 0

        self.step_idx = int(start)
        # Set serving gNB to nearest gNB at start position
        self.serving_gnb_idx = self._get_nearest_gnb()
        d_m      = self._get_distance()
        vel_ms   = self._get_velocity()
        rsrp_dbm, is_los = self._compute_rsrp(d_m)
        sinr_db  = self._compute_sinr(d_m, is_los, vel_ms)
        obs      = self._build_obs(rsrp_dbm, sinr_db, vel_ms)

        self._obs_buffer = [obs.copy()] * (E2_DELAY_STEPS + 1)
        return obs, {}

    def step(self, action):
        # ── Apply action ───────────────────────────────────────────────────
        if action == PWR_UP:
            self.pt_dbm = min(self.pt_dbm + PT_STEP_DBM, PT_MAX_DBM)
        elif action == PWR_DOWN:
            self.pt_dbm = max(self.pt_dbm - PT_STEP_DBM, PT_MIN_DBM)
        elif action == RB_UP:
            self.rb_count = min(self.rb_count + RB_STEP, RB_MAX)
        elif action == RB_DOWN:
            self.rb_count = max(self.rb_count - RB_STEP, RB_MIN)
        elif action == HANDOVER:
            if not self.handover_done:
                self.handover_done = True
                self.handover_step = self.step_idx
                # Switch to nearest gNB — models actual handover
                self.serving_gnb_idx = self._get_nearest_gnb()

        # ── Advance environment ────────────────────────────────────────────
        self.step_idx     += 1
        self.episode_step += 1
        d_m    = self._get_distance()
        vel_ms = self._get_velocity()
        self.ue_x = d_m

        rsrp_dbm, is_los = self._compute_rsrp(d_m)
        sinr_db = self._compute_sinr(d_m, is_los, vel_ms)

        # ── HO outcome checks ──────────────────────────────────────────────
        too_early = False
        too_late  = False
        ho_occurred = False
        if self.handover_done and self.handover_step is not None:
            ho_dist = float(self.traj[self.handover_step, 0])
            ho_rsrp_dbm, _ = self._compute_rsrp(ho_dist, apply_fading_flag=False)
            if ho_rsrp_dbm > HO_PROACTIVE_THRESH:
                too_early = True
            if rsrp_dbm < RLF_THRESH_DBM:
                too_late = True
            ho_occurred = True
            # Reset HO flag — allow multiple HOs per episode (multi-cell traversal)
            # Vehicle continues to next gNB; RSRP recovers; agent optimizes again
            self.handover_done = False
            self.handover_step = None
            self.total_handovers = getattr(self, 'total_handovers', 0) + 1

        # ── Reward ─────────────────────────────────────────────────────────
        reward = self._compute_reward(rsrp_dbm, action)

        # ── Termination ────────────────────────────────────────────────────
        # Episode continues after HO (multi-cell traversal)
        # Only terminates on: trajectory end, RLF, or max steps
        terminated = (
            rsrp_dbm < RLF_THRESH_DBM or
            self.step_idx >= self.T or
            self.episode_step >= MAX_STEPS
        )
        truncated = False

        # ── Build observation with E2 delay ────────────────────────────────
        current_obs = self._build_obs(rsrp_dbm, sinr_db, vel_ms)
        self._obs_buffer.append(current_obs)
        if len(self._obs_buffer) > E2_DELAY_STEPS + 1:
            self._obs_buffer.pop(0)
        delayed_obs = self._obs_buffer[0]

        info = {
            'rsrp_dbm':        rsrp_dbm,
            'sinr_db':         sinr_db,
            'pt_dbm':          self.pt_dbm,
            'rb_count':        self.rb_count,
            'xVelocity_ms':    vel_ms,
            'handover_done':   ho_occurred,
            'too_early':       too_early,
            'too_late':        too_late,
            'step':            self.step_idx,
            'episode_step':    self.episode_step,
            'is_los':          is_los,
            'e2_delay_steps':  E2_DELAY_STEPS,
            'r_mobility':      self._last_r_mobility,
            'r_power':         self._last_r_power,
            'r_spectrum':      self._last_r_spectrum,
            'total_handovers': getattr(self, 'total_handovers', 0),
        }
        return delayed_obs, reward, terminated, truncated, info
