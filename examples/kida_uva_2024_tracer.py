"""
Tracer integration of the bundled KIDA uva 2024 network with interpolation comparison.

Solves the non-autonomous quadratic ODE

    dx/dt = A(p(t)) x + B(p(t))(x ⊗ x)

where x is the vector of species abundances per H nucleus and p(t) is the time-dependent physical environment of a tracer particle — a fluid parcel whose trajectory through an evolving cloud (density, temperature, radiation
field) is recorded by the hydrodynamics simulation. The chemistry is then post-processed along that trajectory by solving the ODE.

Three schemes for interpolating p(t) between hydrodynamic snapshots are compared (piecewise_constant, cubic_spline, pchip). Species are then ranked by the maximum log-space disagreement between schemes to quantify sensitivity
to the interpolation choice. Output is written to examples/data/kida_uva_2024_tracer/.

Run from the repository root or from within examples/:

    python examples/kida_uva_2024_tracer.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import Network, load_abundances
from solver import QuadraticSolverTracer

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "kida.uva.2024" / "gas_reactions_kida.uva.2024.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "kida.uva.2024" / "abundances.in"
SAVE_DIR = HERE / "data" / "kida_uva_2024_tracer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR      = 3600 * 24 * 365.25

# Tolerances for a scaled solve
ATOL      = 1e-6
RTOL      = 1e-3

MIN_SCALE = 1e-22
N_PLOT    = 12
LOG_COLS  = ["nH", "uv_flux", "Av"]
METHOD    = "BDF"

# ---------------------------------------------------------------------------
# Load network
# ---------------------------------------------------------------------------

print("Loading KIDA uva 2024 network...")
net = Network(grains=True)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

abund = load_abundances(str(ABUNDANCES_PATH))
x0 = np.zeros(len(net.species), dtype=np.float64)
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val

print("\nNetwork")
print(f"  path      = {NETWORK_PATH}")
print(f"  species   = {len(net.species)}")
print(f"  reactions = {len(net.reactions)}")
print(f"  dropped passive = {dropped}")
print(f"  nonzero x0 = {int(np.count_nonzero(x0))}")

# ---------------------------------------------------------------------------
# Synthetic tracer trajectory
# ---------------------------------------------------------------------------

t_kyr    = np.linspace(0.0, 1600.0, 25)
dt_hydro = (t_kyr[1] - t_kyr[0]) * 1e3 * YEAR

nH = (
    80.0
    + 2500.0 / (1.0 + np.exp(-(t_kyr - 700.0) / 120.0))
    + 1800.0 * np.exp(-0.5 * ((t_kyr - 950.0) / 220.0) ** 2)
)
T = (
    80.0
    + 800.0 * np.exp(-0.5 * ((t_kyr - 420.0) / 180.0) ** 2)
    + 4300.0 * np.exp(-0.5 * ((t_kyr - 900.0) / 170.0) ** 2)
    + 600.0 / (1.0 + np.exp(-(t_kyr - 1200.0) / 120.0))
)
Tgrain = (
    17.0
    - 2.2 * (t_kyr / t_kyr.max())
    + 1.7 * np.exp(-0.5 * ((t_kyr - 760.0) / 260.0) ** 2)
)
Av = (
    0.65
    + 0.75 * np.exp(-0.5 * ((t_kyr - 620.0) / 200.0) ** 2)
    - 0.58 * (t_kyr / t_kyr.max())
)
uv_flux = (
    120.0 * np.exp(-4.2 * Av)
    + 18.0 * np.exp(-0.5 * ((t_kyr - 1350.0) / 170.0) ** 2)
    + 1.0e-5
)

pt = np.column_stack([nH, T, Tgrain, Av, uv_flux]).astype(np.float64)

print("\nTracer trajectory")
print(f"  hydro steps = {pt.shape[0] - 1}")
print(f"  dt_hydro    = {dt_hydro / YEAR / 1e3:.1f} kyr")
print(f"  total time  = {(pt.shape[0] - 1) * dt_hydro / YEAR / 1e6:.3f} Myr")

# ---------------------------------------------------------------------------
# Plot physical parameter profiles — one curve per interpolation mode
# ---------------------------------------------------------------------------

colors     = {"piecewise_constant": "tab:blue", "cubic_spline": "tab:orange", "pchip": "tab:green"}
linestyles = {"piecewise_constant": "-",         "cubic_spline": "--",         "pchip": ":"}

_PT_COLS  = ("nH", "T", "Tgrain", "Av", "uv_flux")
_log_mask = np.array([c in set(LOG_COLS) for c in _PT_COLS])
t_knots   = np.arange(pt.shape[0], dtype=np.float64) * dt_hydro
t_fine    = np.linspace(t_knots[0], t_knots[-1], 500)

_pt_fit = pt.copy()
_pt_fit[:, _log_mask] = np.log(_pt_fit[:, _log_mask])

_cs_interp = CubicSpline(t_knots, _pt_fit, axis=0, extrapolate=False)
_ph_interp = PchipInterpolator(t_knots, _pt_fit, axis=0, extrapolate=False)

def _eval_spline(interp, t_arr):
    vals = np.asarray(interp(t_arr), dtype=np.float64).copy()
    vals[:, _log_mask] = np.exp(vals[:, _log_mask])
    return vals

def _eval_pc(t_arr):
    idx = np.clip(np.searchsorted(t_knots, t_arr, side="left") - 1, 0, pt.shape[0] - 1)
    return pt[idx]

param_curves = {
    "piecewise_constant": _eval_pc(t_fine),
    "cubic_spline":       _eval_spline(_cs_interp, t_fine),
    "pchip":              _eval_spline(_ph_interp, t_fine),
}

t_fine_kyr = t_fine / (1e3 * YEAR)

parameter_plot = SAVE_DIR / "kida_uva_2024_parameter_profiles.pdf"
fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)

profile_names  = ["nH", "T", "Tgrain", "Av", "uv_flux"]
profile_scales = ["log", "linear", "linear", "log", "log"]

for ax, name, scale, col_idx in zip(axes, profile_names, profile_scales, range(5)):
    ax.plot(t_kyr, pt[:, col_idx], "o", color="black", markersize=4, zorder=3)
    for mode in ("piecewise_constant", "cubic_spline", "pchip"):
        ax.plot(t_fine_kyr, param_curves[mode][:, col_idx],
                color=colors[mode], linestyle=linestyles[mode], linewidth=1.5, label=mode)
    ax.set_ylabel(name)
    ax.set_yscale(scale)
    ax.grid(True, alpha=0.3)

axes[0].legend(frameon=False, ncol=3)
axes[-1].set_xlabel("Time (kyr)")
fig.tight_layout()
fig.savefig(parameter_plot, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Tracer solves — three interpolation modes
# ---------------------------------------------------------------------------

tracer = QuadraticSolverTracer()
tracer_data     = {}
raw_tracer_output = {}

print("\nRunning tracer solves...")
for mode in ("piecewise_constant", "cubic_spline", "pchip"):
    out_t, out_y = tracer.solve(
        dt_hydro=dt_hydro,
        pt=pt,
        get_tensors=net.get_operators,
        x0=x0,
        atol=ATOL,
        rtol=RTOL,
        min_scale=MIN_SCALE,
        method=METHOD,
        interpolation=mode,
        use_scaling=True,
        log_cols=LOG_COLS,
    )
    raw_tracer_output[mode] = (out_t, out_y)

    if mode == "piecewise_constant":
        t_mode = [np.asarray(out_t[0], dtype=np.float64)]
        y_mode = [np.asarray(out_y[0], dtype=np.float64)]
        for i in range(1, len(out_t)):
            t_mode.append(np.asarray(out_t[i], dtype=np.float64)[1:])
            y_mode.append(np.asarray(out_y[i], dtype=np.float64)[:, 1:])
        t_mode = np.concatenate(t_mode)
        y_mode = np.hstack(y_mode)
    else:
        t_mode = np.asarray(out_t, dtype=np.float64)
        y_mode = np.asarray(out_y, dtype=np.float64)

    tracer_data[mode] = (t_mode, y_mode)

    print(f"  {mode}")
    print(f"    output shape = {y_mode.shape}")
    print(f"    finite = {bool(np.isfinite(y_mode).all())}")
    print(f"    final min/max = ({float(y_mode[:, -1].min()):.6e}, {float(y_mode[:, -1].max()):.6e})")

# ---------------------------------------------------------------------------
# Rank species by maximum log-space disagreement between modes
# ---------------------------------------------------------------------------

y_pc = tracer_data["piecewise_constant"][1][:, -1]
y_cs = tracer_data["cubic_spline"][1][:, -1]
y_ph = tracer_data["pchip"][1][:, -1]

log_pc = np.log10(np.maximum(y_pc, MIN_SCALE))
log_cs = np.log10(np.maximum(y_cs, MIN_SCALE))
log_ph = np.log10(np.maximum(y_ph, MIN_SCALE))

err_cs_vs_pc = np.abs(log_cs - log_pc)
err_ph_vs_pc = np.abs(log_ph - log_pc)
err_cs_vs_ph = np.abs(log_cs - log_ph)

max_log_error = np.maximum.reduce([err_cs_vs_pc, err_ph_vs_pc, err_cs_vs_ph])
ranked_idx    = np.argsort(max_log_error)[::-1]
top_idx       = ranked_idx[: min(N_PLOT, len(ranked_idx))]
top_species   = [net.species[i] for i in top_idx]

print("\nLargest interpolation differences")
for i in top_idx:
    print(f"  {net.species[i]}")
    print(f"    max log10 error    = {max_log_error[i]:.6e} dex")
    print(f"    cubic vs piecewise = {err_cs_vs_pc[i]:.6e} dex")
    print(f"    pchip vs piecewise = {err_ph_vs_pc[i]:.6e} dex")
    print(f"    cubic vs pchip     = {err_cs_vs_ph[i]:.6e} dex")

# ---------------------------------------------------------------------------
# Save ranking table
# ---------------------------------------------------------------------------

ranking_txt = SAVE_DIR / "kida_uva_2024_interpolation_ranking.txt"
with open(ranking_txt, "w", encoding="ascii") as f:
    f.write("species max_log10_error_dex cubic_vs_piecewise_dex pchip_vs_piecewise_dex cubic_vs_pchip_dex\n")
    for i in ranked_idx:
        f.write(
            f"{net.species[i]} "
            f"{max_log_error[i]:.12e} "
            f"{err_cs_vs_pc[i]:.12e} "
            f"{err_ph_vs_pc[i]:.12e} "
            f"{err_cs_vs_ph[i]:.12e}\n"
        )

# ---------------------------------------------------------------------------
# Plot worst species across all three modes
# ---------------------------------------------------------------------------

colors     = {"piecewise_constant": "tab:blue", "cubic_spline": "tab:orange", "pchip": "tab:green"}
linestyles = {"piecewise_constant": "-",         "cubic_spline": "--",         "pchip": ":"}
plot_order = ("piecewise_constant", "cubic_spline", "pchip")

comparison_pdf = SAVE_DIR / "kida_uva_2024_interpolation_errors.pdf"
fig, axes = plt.subplots(len(top_species), 1, figsize=(8, 2.6 * len(top_species)), sharex=True)
if len(top_species) == 1:
    axes = [axes]

for ax, species_name in zip(axes, top_species):
    idx = net.species_map[species_name]
    for mode in plot_order:
        t_mode, y_mode = tracer_data[mode]
        ax.plot(
            t_mode / YEAR,
            y_mode[idx],
            color=colors[mode],
            linestyle=linestyles[mode],
            linewidth=1.5,
            label=mode,
            zorder=1 if mode == "piecewise_constant" else 2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(species_name)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{species_name}  |  max log10 error = {max_log_error[idx]:.3e} dex", fontsize=10)

axes[0].legend(frameon=False, ncol=3, loc="upper left")
axes[-1].set_xlabel("Time (years)")
fig.tight_layout()
fig.savefig(comparison_pdf, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Save tracer output for each mode
# ---------------------------------------------------------------------------

for mode in ("piecewise_constant", "cubic_spline", "pchip"):
    out_root = SAVE_DIR / f"kida_uva_2024_tracer_{mode}"
    out_t, out_y = raw_tracer_output[mode]
    if mode == "piecewise_constant":
        tracer.save_data(out_t, out_y, pt=pt, species=net.species,
                         save_path=str(out_root), save_csv=True, interpolation=mode)
    else:
        tracer.save_data(out_t, out_y, pt=pt, species=net.species,
                         save_path=str(out_root), save_csv=True,
                         dt_hydro=dt_hydro, interpolation=mode, log_cols=LOG_COLS)

print(f"\nRanking saved to:  {ranking_txt}")
print(f"Error plot saved to: {comparison_pdf}")
print(f"\nAll output written to: {SAVE_DIR}")
