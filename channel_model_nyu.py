"""
channel_model_nyu.py
--------------------
Measurement-calibrated large-scale channel for the 140 GHz Sub-THz env, replacing the
generic free-space (n=2, no shadow, no array gain) path loss with the NYU **close-in (CI)
free-space reference-distance path loss model** measured at 142 GHz in an urban microcell
(UMi), PLUS log-normal shadow fading and a directional (beamforming) array-gain term.

WHAT THIS IS (be precise):
  - The CI *large-scale path-loss* model that NYUSIM uses, with the ACTUAL measured
    142 GHz UMi parameters. It is measurement-CALIBRATED (real PLE + shadow-fading std),
    not a real captured channel, and not the full NYUSIM spatial (cluster/subpath) model.
  - Small-scale fading is left to the existing alpha-mu model (channel_model.apply_fading);
    this module only changes the deterministic path loss + adds shadow fading + array gain.

MEASURED PARAMETERS (omnidirectional CI, d0 = 1 m), 142 GHz downtown-Brooklyn UMi campaign:
    LOS : n = 1.9, sigma_SF = 2.7 dB
    NLOS: n = 2.9, sigma_SF = 8.2 dB
  Source: Xing, Ju & Rappaport, "Propagation Measurements and Path Loss Models for
  sub-THz in Urban Microcells" (140/142 GHz), NYU WIRELESS. (Omnidirectional PLE/sigma;
  the directional values n=2.1/3.1, sigma=2.8/8.3 are NOT used here because antenna
  directivity is modelled separately by the array-gain term below.)
  PDF: https://par.nsf.gov/servlets/purl/10309452 ; arXiv:2110.06361 (142 GHz UMi model).

ARRAY / BEAMFORMING GAIN (stated modelling assumption, NOT a measured value):
  ARRAY_GAIN_DBI is the combined boresight gain of the (aligned) serving beam. At 140 GHz
  a directional array is required to close the link; the value is an explicit design
  assumption and is calibrated in nyu_channel_impact_v6.py to keep the scenario coherent.
  Co-channel interferers are assumed beam-MISALIGNED -> sidelobe gain
  INTERFERER_ARRAY_GAIN_DBI (default 0 dBi), which is how beamforming suppresses interference.
"""
import numpy as np

# ── Physical constants ──────────────────────────────────────────────────────
C     = 3e8
FREQ  = 140e9
D0    = 1.0          # CI reference distance (m)

# ── Measured 142 GHz UMi CI parameters (omnidirectional) ────────────────────
N_LOS,  SIG_LOS  = 1.9, 2.7    # path-loss exponent, shadow-fading std (dB), LOS
N_NLOS, SIG_NLOS = 2.9, 8.2    # path-loss exponent, shadow-fading std (dB), NLOS

# ── Array / beamforming gain (STATED ASSUMPTION; calibrated in impact script) ─
ARRAY_GAIN_DBI            = 15.0   # combined boresight gain of the aligned serving beam
INTERFERER_ARRAY_GAIN_DBI = 0.0   # misaligned interferer beams -> sidelobe (0 dBi)


def fspl_d0_db(freq=FREQ, d0=D0):
    """Free-space path loss at the CI reference distance d0 (dB)."""
    return 20.0 * np.log10(4.0 * np.pi * freq * d0 / C)


def ci_pathloss_db(d_m, is_los, freq=FREQ):
    """
    Mean close-in (CI) path loss at distance d_m (dB), WITHOUT shadow fading.
        PL_CI(d) = FSPL(d0) + 10 * n * log10(d / d0)
    n is the measured LOS/NLOS PLE. Shadow fading (log-normal, sigma below) is added
    separately by the environment, per serving-link association, not here.
    """
    d = np.maximum(np.asarray(d_m, dtype=float), D0)   # CI is defined for d >= d0
    n = N_LOS if is_los else N_NLOS
    return fspl_d0_db(freq) + 10.0 * n * np.log10(d / D0)


def sigma_sf(is_los):
    """Shadow-fading standard deviation (dB) for the LOS/NLOS regime."""
    return SIG_LOS if is_los else SIG_NLOS


# ── Self-test / sanity checks ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=== NYU 142 GHz CI channel self-test ===\n")
    pl0 = fspl_d0_db()
    print(f"FSPL(d0=1 m) at 140 GHz = {pl0:.2f} dB  (expected ~75.36)")
    assert abs(pl0 - 75.36) < 0.05, "FSPL(1m) check failed"

    # CI with n=2, no shadow must equal generic free-space 20log10(4*pi*f*d/c)
    for d in [5.0, 10.0, 38.7, 100.0]:
        ci_n2 = fspl_d0_db() + 10*2.0*np.log10(d/D0)
        fs    = 20*np.log10(4*np.pi*FREQ*d/C)
        assert abs(ci_n2 - fs) < 1e-6, "CI(n=2) != free-space"
    print("CI(n=2, no shadow) == free-space model  [PASS]\n")

    print(f"{'d (m)':>7} | {'FS n=2':>8} | {'CI LOS n=1.9':>12} | {'CI NLOS n=2.9':>13}")
    print("  " + "-"*46)
    for d in [10.0, 20.0, 37.4, 38.7, 74.8, 100.0]:
        fs   = 20*np.log10(4*np.pi*FREQ*d/C)
        clos = ci_pathloss_db(d, True)
        cnl  = ci_pathloss_db(d, False)
        print(f"{d:>7.1f} | {fs:>8.1f} | {clos:>12.1f} | {cnl:>13.1f}")
    print("\nShadow-fading std: LOS +/-{} dB, NLOS +/-{} dB".format(SIG_LOS, SIG_NLOS))
    print("=== self-test complete ===")
