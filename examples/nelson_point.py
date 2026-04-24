"""
Point integration of the Nelson gas-phase network.

Solves the autonomous quadratic ODE

    dx/dt = A x + B(x ⊗ x)

where x is the vector of species abundances per H nucleus. A (N×N) encodes unimolecular reaction rate coefficients and B (N×N²) encodes bimolecular reaction rate coefficients. Both matrices are assembled once from the reaction network
evaluated at a single fixed physical environment (gas temperature, H number density, visual extinction, UV field) and held constant throughout the integration.

Output is written to examples/data/nelson_point/.

Run from the repository root or from within examples/:

    python examples/nelson_point.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import Network, load_abundances
from solver import QuadraticSolver

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"
SAVE_DIR = HERE / "data" / "nelson_point"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3

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
# Environment
# ---------------------------------------------------------------------------

env = dict(
    T       = 10.0,    # gas temperature [K]
    nH      = 2e4,     # total H number density [cm⁻³]
    Av      = 15.0,    # visual extinction [mag]
    uv_flux = 1.0,     # FUV field scaling (1 = standard Draine field)
)

A, B = net.get_operators(env)

# ---------------------------------------------------------------------------
# Integrate
# ---------------------------------------------------------------------------

t_eval = np.logspace(0.0, np.log10(1e6 * YEAR), 120)

solver = QuadraticSolver()
t, y = solver.solve(
    A, B,
    t_span=(t_eval[0], t_eval[-1]),
    x0=x0,
    atol=ATOL,
    rtol=RTOL,
    t_eval=t_eval,
)

print("\nSolve")
print(f"  A shape = {A.shape}, nnz = {A.nnz}")
print(f"  B shape = {B.shape}, nnz = {B.nnz}")
print(f"  output shape = {y.shape}")
print(f"  finite = {bool(np.isfinite(y).all())}")
print(f"  final min/max = ({float(y[:, -1].min()):.6e}, {float(y[:, -1].max()):.6e})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_root = str(SAVE_DIR / "nelson_point")
solver.save(out_root, t, y, col_names=net.species)
print(f"\n  saved = {out_root}.npy")
print(f"  saved = {out_root}.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plot_species = list(net.species)

fig, ax = solver.plot(
    t / YEAR, y,
    indices=[net.species_map[s] for s in plot_species],
    labels=plot_species,
)
ax.set_xlabel("Time (years)")
ax.set_ylabel("Abundance per H")

out_pdf = str(SAVE_DIR / "nelson_point.pdf")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"  plot   = {out_pdf}")
