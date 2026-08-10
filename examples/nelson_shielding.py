"""
Effect of CO self-shielding on the Nelson gas-phase network.

Integrates the point model to chemical steady state over a grid of visual extinctions, once with ``self_shielding=False`` and once with ``self_shielding=True``, and compares the resulting carbon partitioning.

The Lee et al. (1996) shielding factors multiply the FUV photodissociation rates:

    k(CO) = alpha * uv_flux * exp(-gamma*Av) * theta_CO(N_CO) * theta_H2CO(N_H2)

Only the CO channel is affected here. The Nelson network has no H2 photodissociation reaction at all (H2 is treated as fully self-shielded), so ``get_shielded_reactions()`` returns an empty H2 list and theta_H2 never enters the ODE. It is still plotted for reference, since it is what would shield H2 in a network that photodissociates it.

Both networks are built with ``dust_attenuation=True`` so that Av enters the unshielded rate through the ``exp(-gamma*Av)`` continuum term as well. Without it, ``uv_flux`` is taken to be pre-attenuated and Av would act only through the shielding factors.

Output is written to examples/data/nelson_shielding/.

Run from the repository root or from within examples/:

    python examples/nelson_shielding.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shielding
from parser import Network, load_abundances
from solver import QuadraticSolver

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
NETWORK_PATH    = REPO_ROOT / "networks" / "nelson" / "gas_reactions.in"
ABUNDANCES_PATH = REPO_ROOT / "networks" / "nelson" / "abundances.in"
SAVE_DIR = HERE / "data" / "nelson_shielding"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 3600 * 24 * 365.25
ATOL = 1e-20
RTOL = 1e-3
T_END = 1e6 * YEAR                       # long enough for steady state at these densities

AV_GRID = np.logspace(np.log10(0.1), np.log10(20.0), 15)
ENV_FIXED = dict(T=30.0, nH=1e3, uv_flux=1.0)   # low Av needs a warmer, thinner cloud edge

TRACK = ("CO", "C", "C+")                # carbon partitioning is what shielding moves

# ---------------------------------------------------------------------------
# Load network, once per shielding setting
# ---------------------------------------------------------------------------


def build(self_shielding):
    net = Network(grains=False, self_shielding=self_shielding, dust_attenuation=True)
    net.load_from_disk(str(NETWORK_PATH))
    net.drop_passive_species()
    return net


print("Loading Nelson network...")
nets = {"unshielded": build(False), "shielded": build(True)}
net = nets["shielded"]

shielded_rxns = net.get_shielded_reactions()
print("\nNetwork")
print(f"  path      = {NETWORK_PATH}")
print(f"  species   = {len(net.species)}")
print(f"  reactions = {len(net.reactions)}")
print(f"  shielded channels = {shielded_rxns}")
for ch, idxs in shielded_rxns.items():
    for i in idxs:
        r = net.reactions[i]
        print(f"    {ch}: {' + '.join(r['reactants'])} -> {' + '.join(r['products'])}"
              f"   (alpha={r['alpha']:.3e}, gamma={r['gamma']:.3g})")
if not shielded_rxns["H2"]:
    print("    H2: none in this network (H2 is not photodissociated)")

# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

abund = load_abundances(str(ABUNDANCES_PATH))
abund["e-"] = sum(val for name, val in abund.items() if name.endswith("+"))

x0 = np.zeros(len(net.species), dtype=np.float64)
for name, val in abund.items():
    if name in net.species_map:
        x0[net.species_map[name]] = val
print(f"  nonzero x0 = {int(np.count_nonzero(x0))}")

# ---------------------------------------------------------------------------
# Shielding factors and column densities over the Av grid
# ---------------------------------------------------------------------------

N_H2, N_CO = shielding.column_densities(AV_GRID)
f_H2, f_CO = shielding.shield_factors(AV_GRID)

print("\nShielding factors")
print(f"  {'Av':>7}  {'N(H2)':>10}  {'N(CO)':>10}  {'theta_H2':>10}  {'theta_CO':>10}")
for k, av in enumerate(AV_GRID):
    print(f"  {av:7.3f}  {N_H2[k]:10.3e}  {N_CO[k]:10.3e}"
          f"  {f_H2[k]:10.3e}  {f_CO[k]:10.3e}")

# ---------------------------------------------------------------------------
# Integrate to steady state at each Av, with and without shielding
# ---------------------------------------------------------------------------

solver = QuadraticSolver()
final = {key: np.zeros((AV_GRID.size, len(net.species))) for key in nets}

print("\nSolving")
for key, n in nets.items():
    for k, av in enumerate(AV_GRID):
        A, B = n.get_operators(dict(**ENV_FIXED, Av=float(av)))
        _, y = solver.solve(A, B, t_span=(0.0, T_END), x0=x0, atol=ATOL, rtol=RTOL)
        final[key][k] = y[:, -1]
    print(f"  {key:11s} done ({AV_GRID.size} solves), finite = "
          f"{bool(np.isfinite(final[key]).all())}")

idx = {s: net.species_map[s] for s in TRACK}
print(f"\n  {'Av':>7}  " + "  ".join(f"{s+' off':>11} {s+' on':>11}" for s in TRACK))
for k, av in enumerate(AV_GRID):
    cells = []
    for s in TRACK:
        cells.append(f"{final['unshielded'][k, idx[s]]:11.4e} "
                     f"{final['shielded'][k, idx[s]]:11.4e}")
    print(f"  {av:7.3f}  " + "  ".join(cells))

co_off = final["unshielded"][:, idx["CO"]]
co_on = final["shielded"][:, idx["CO"]]
ratio = co_on / np.maximum(co_off, 1e-300)
print(f"\n  CO enhancement from shielding: min={ratio.min():.3f}x  max={ratio.max():.3f}x"
      f"  (at Av={AV_GRID[int(np.argmax(ratio))]:.3f})")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

header = (["Av", "N_H2", "N_CO", "theta_H2", "theta_CO"]
          + [f"{s}_unshielded" for s in net.species]
          + [f"{s}_shielded" for s in net.species])
out = np.hstack([AV_GRID[:, None], N_H2[:, None], N_CO[:, None],
                 f_H2[:, None], f_CO[:, None],
                 final["unshielded"], final["shielded"]])

out_root = str(SAVE_DIR / "nelson_shielding")
np.save(out_root + ".npy", out)
np.savetxt(out_root + ".csv", out, delimiter=",", header=",".join(header), comments="")
print(f"\n  saved = {out_root}.npy")
print(f"  saved = {out_root}.csv")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)

ax0.plot(AV_GRID, f_CO, "-o", ms=3, label=r"$\theta_{\rm CO}$ (applied)")
ax0.plot(AV_GRID, f_H2, "--s", ms=3, label=r"$\theta_{\rm H_2}$ (unused here)")
ax0.set_xscale("log")
ax0.set_yscale("log")
ax0.set_ylabel("shielding factor")
ax0.grid(True, which="both", alpha=0.3)
ax0.legend(frameon=False)
ax0.set_title("Lee et al. (1996) shielding factors")

cmap = plt.get_cmap("tab10")
for i, s in enumerate(TRACK):
    ax1.plot(AV_GRID, final["unshielded"][:, idx[s]], "--", color=cmap(i),
             label=f"{s}, shielding off")
    ax1.plot(AV_GRID, final["shielded"][:, idx[s]], "-o", ms=3, color=cmap(i),
             label=f"{s}, shielding on")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$A_V$ (mag)")
ax1.set_ylabel("steady-state abundance per H")
ax1.grid(True, which="both", alpha=0.3)
ax1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
ax1.set_title(f"T = {ENV_FIXED['T']:g} K, "
              rf"$n_{{\rm H}}$ = {ENV_FIXED['nH']:g} cm$^{{-3}}$")

fig.subplots_adjust(right=0.72)
out_pdf = str(SAVE_DIR / "nelson_shielding.pdf")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)
print(f"  plot   = {out_pdf}")
