"""
Tracer integration of the Nelson gas-phase network with interpolation comparison.

Solves the non-autonomous quadratic ODE

    dx/dt = A(p(t)) x + B(p(t))(x ⊗ x)

where x is the vector of species abundances per H nucleus and p(t) is the time-dependent physical environment of a tracer particle — a fluid parcel whose trajectory through an evolving cloud (density, temperature, radiation
field) is recorded by the hydrodynamics simulation. The chemistry is then post-processed along that trajectory by solving the ODE.

Three schemes for interpolating p(t) between hydrodynamic snapshots are compared (piecewise_constant, cubic_spline, pchip). Output is written to examples/data/nelson_tracer/.

Run from the repository root or from within examples/:

    python examples/nelson_tracer.py
"""

import sys
from pathlib import Path
from scipy.interpolate import CubicSpline, PchipInterpolator

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import Network, load_abundances
from solver import QuadraticSolverTracer

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"
SAVE_DIR = HERE / "data" / "nelson_tracer"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR      = 3600 * 24 * 365.25

# Tolerances for a scaled solve
ATOL      = 1e-6
RTOL      = 1e-3

MIN_SCALE = 1e-20
LOG_COLS  = ["nH", "uv_flux", "Av"]

# ---------------------------------------------------------------------------
# Load network
# ---------------------------------------------------------------------------

print("Loading Nelson network...")
net = Network(grains=False)
net.load_from_disk(str(NETWORK_PATH))
dropped = net.drop_passive_species()

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

abund = load_abundances(str(ABUNDANCES_PATH))
abund["e-"] = sum(val for name, val in abund.items() if name.endswith("+"))

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
# UV field weakens as shielding builds up and partially recovers as Av drops.
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

parameter_plot = SAVE_DIR / "nelson_parameter_profiles.pdf"
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
tracer_plot_data = {}

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
        interpolation=mode,
        use_scaling=True,
        log_cols=LOG_COLS,
    )

    tracer_root = SAVE_DIR / f"nelson_tracer_{mode}"

    if mode == "piecewise_constant":
        tracer.save_data(out_t, out_y, pt=pt, species=net.species,
                         save_path=str(tracer_root), save_csv=True, interpolation=mode)

        t_plot = [np.asarray(out_t[0], dtype=np.float64)]
        y_plot = [np.asarray(out_y[0], dtype=np.float64)]
        for i in range(1, len(out_t)):
            t_plot.append(np.asarray(out_t[i], dtype=np.float64)[1:])
            y_plot.append(np.asarray(out_y[i], dtype=np.float64)[:, 1:])
        t_plot = np.concatenate(t_plot)
        y_plot = np.hstack(y_plot)
        tracer_plot_data[mode] = (t_plot, y_plot)

        final_state = out_y[-1][:, -1]
        finite = all(np.isfinite(chunk).all() for chunk in out_y)

        print(f"\nTracer [{mode}]")
        print(f"  segments = {len(out_t)}")
        print(f"  total output points = {sum(len(chunk) for chunk in out_t)}")
        print(f"  finite = {finite}")
        print(f"  final min/max = ({float(final_state.min()):.6e}, {float(final_state.max()):.6e})")
        print(f"  saved = {tracer_root}.npy, {tracer_root}.csv")

    else:
        tracer.save_data(out_t, out_y, pt=pt, species=net.species,
                         save_path=str(tracer_root), save_csv=True,
                         dt_hydro=dt_hydro, interpolation=mode, log_cols=LOG_COLS)

        tracer_plot_data[mode] = (out_t, out_y)

        print(f"\nTracer [{mode}]")
        print(f"  output shape = {out_y.shape}")
        print(f"  finite = {bool(np.isfinite(out_y).all())}")
        print(f"  final min/max = ({float(out_y[:, -1].min()):.6e}, {float(out_y[:, -1].max()):.6e})")
        print(f"  saved = {tracer_root}.npy, {tracer_root}.csv")

# ---------------------------------------------------------------------------
# Plot all species across interpolation modes
# ---------------------------------------------------------------------------

colors     = {"piecewise_constant": "tab:blue", "cubic_spline": "tab:orange", "pchip": "tab:green"}
linestyles = {"piecewise_constant": "-",         "cubic_spline": "--",         "pchip": ":"}
plot_order = ("piecewise_constant", "cubic_spline", "pchip")

plot_species = list(net.species)
tracer_plot  = SAVE_DIR / "nelson_tracer_comparison.pdf"

fig, axes = plt.subplots(len(plot_species), 1, figsize=(7, 3 * len(plot_species)), sharex=True)
if len(plot_species) == 1:
    axes = [axes]

for ax, species_name in zip(axes, plot_species):
    idx = net.species_map[species_name]
    for mode in plot_order:
        t_mode, y_mode = tracer_plot_data[mode]
        ax.plot(
            t_mode / YEAR,
            y_mode[idx],
            label=mode,
            color=colors[mode],
            linestyle=linestyles[mode],
            linewidth=1.8,
            zorder=1 if mode == "piecewise_constant" else 2,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(species_name)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (years)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(tracer_plot, bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------------
# Numerical comparison between interpolation modes
# ---------------------------------------------------------------------------

comparison_rows = []
print("\nInterpolation comparison (final abundances)")
for species_name in plot_species:
    idx      = net.species_map[species_name]
    final_pc = tracer_plot_data["piecewise_constant"][1][idx, -1]
    final_cs = tracer_plot_data["cubic_spline"][1][idx, -1]
    final_ph = tracer_plot_data["pchip"][1][idx, -1]

    rel_cs    = abs(final_cs - final_pc) / max(abs(final_pc), 1e-300)
    rel_ph    = abs(final_ph - final_pc) / max(abs(final_pc), 1e-300)
    rel_cs_ph = abs(final_cs - final_ph) / max(abs(final_ph), 1e-300)

    comparison_rows.append([species_name, final_pc, final_cs, final_ph,
                             rel_cs, rel_ph, rel_cs_ph])

    print(f"  {species_name}")
    print(f"    piecewise_constant = {final_pc:.6e}")
    print(f"    cubic_spline       = {final_cs:.6e}   rel. diff vs piecewise = {rel_cs:.6e}")
    print(f"    pchip              = {final_ph:.6e}   rel. diff vs piecewise = {rel_ph:.6e}")
    print(f"    cubic vs pchip rel. diff = {rel_cs_ph:.6e}")

comparison_txt = SAVE_DIR / "nelson_tracer_comparison.txt"
with open(comparison_txt, "w", encoding="ascii") as f:
    f.write("species piecewise_constant cubic_spline pchip "
            "rel_cs_vs_pc rel_pchip_vs_pc rel_cs_vs_pchip\n")
    for row in comparison_rows:
        f.write(
            f"{row[0]} {row[1]:.12e} {row[2]:.12e} {row[3]:.12e} "
            f"{row[4]:.12e} {row[5]:.12e} {row[6]:.12e}\n"
        )

print(f"\nTracer comparison plot: {tracer_plot}")
print(f"Interpolation comparison: {comparison_txt}")
print(f"\nAll output written to: {SAVE_DIR}")
