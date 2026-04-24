"""
Gas-phase + pseudo-grain reaction network parser.

Parses a KIDA-format ``reactions.dat`` file and builds the sparse polynomial
ODE tensors ``A`` and ``B`` for the system

    dx/dt = A x + B(x ⊗ x)

where ``x`` is a vector of species abundances per H nucleus.

Supported chemistry
-------------------
* **Gas-phase reactions** — formula types 1–5 (cosmic-ray ionisation,
  UV photodissociation, Kooij/ionpol rate laws).
* **Pseudo-grain H₂ formation** (``grains=True``) — H₂ formation on grain
  surfaces via the intermediate pseudo-species XH:

    H  →  XH             (frml 11, adsorption pseudo-rate)
    XH + XH  →  H₂ + H   (frml 10, recombination)

* **Ion–grain recombination** (``grains=True``) — ion recombination with
  grains (frml 0, itype 0), involving the pseudo-species GRAIN- and GRAIN0.

Full grain-surface chemistry (adsorption/desorption, surface reactions) is
not supported by this module.

Quick start
-----------
    from gas_parser import Network, load_abundances

    net = Network(grains=True)
    net.load_from_disk("reactions.dat")

    env = dict(T=10.0, nH=1e4, Av=1.0, uv_flux=1.0)  # uv_flux: 1 = standard Draine field
    A, B = net.get_operators(env)

References
----------
Wakelam, V. et al. (2012). A Kinetic Database for Astrochemistry (KIDA).
    ApJS, 199, 21. https://doi.org/10.1088/0067-0049/199/1/21

Wakelam, V. et al. (2024). The 2024 KIDA network for interstellar chemistry.
    A&A, 689, A63. https://doi.org/10.1051/0004-6361/202450606
"""

__all__ = ["Network", "load_abundances"]

import os
import warnings
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from typing import Dict


# ---------------------------------------------------------------------------
# Chemistry constants
# ---------------------------------------------------------------------------

#: Grain number density = grain_gas_ratio * nH
grain_gas_ratio: float = 1.8e-12

#: Cosmic-ray ionisation rate [s⁻¹]
zeta_cr: float = 1.3e-17


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def load_abundances(path: str) -> Dict[str, float]:
    """
    Load initial abundances from a two-column text file.

    Lines starting with ``#`` are treated as comments.  Each data line has
    the form::

        <species>  <abundance>

    The electron abundance ``e-`` must **not** appear in the file; it is
    computed externally from charge neutrality.

    Parameters
    ----------
    path : str
        Path to the abundances file.

    Returns
    -------
    dict
        Mapping ``species_name -> float``.
    """
    abund: Dict[str, float] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            abund[parts[0]] = float(parts[1])
    return abund


# ---------------------------------------------------------------------------
# Pseudo-grain species that are excluded when grains=False
# ---------------------------------------------------------------------------

_PSEUDO_GRAIN_SPECIES = frozenset({"XH", "GRAIN-", "GRAIN0"})


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Network:
    """
    Gas-phase + pseudo-grain reaction network.

    Reads a KIDA-format reactions file and builds sparse ODE tensors A (N×N)
    and B (N×N²) such that

        dx/dt = A x + B(x ⊗ x)

    where x is the vector of species abundances per H nucleus and nH has been
    absorbed into the 2-body rate coefficients.  Column j*N+k of B encodes
    the reactant pair (j, k).

    Parameters
    ----------
    grains : bool
        False (default) — pure gas-phase network.  Reactions with frml 10–11
        and frml 0/itype 0 return zero rates; XH, GRAIN-, and GRAIN0 are
        excluded from the species list.

        True — gas-phase plus pseudo-grain reactions.  H₂ formation via XH
        (frml 10/11) and ion–grain recombination via GRAIN-/GRAIN0 (frml 0,
        itype 0) are active.  Grain temperature is not required.

    Environment dict
    ----------------
    Required keys for get_operators::

        env = dict(
            T        = 10.0,   # gas temperature [K]
            nH       = 1e4,    # total H number density [cm⁻³]
            Av       = 1.0,    # visual extinction [mag]
            uv_flux  = 1.0,    # FUV field scaling (1 = standard Draine field)
        )

    Optional: Tcap_2body (bool, default True) — clamp T to [Tmin, Tmax] when
    evaluating Kooij / ionpol rate coefficients.
    """

    def __init__(self, grains: bool = False):
        """
        Parameters
        ----------
        grains : bool
            Activate pseudo-grain reactions (H₂ formation via XH and
            ion–grain recombination via GRAIN-/GRAIN0).
        """
        self.grains: bool = grains

        self.species: list = []
        self.species_map: dict = {}
        self.reactions: list = []

        # Fields treated as external forcing (not ODE variables)
        self.external_fields = frozenset({"Photon", "CR", "CRP"})

        self.max_order: int = 0
        self._warned_high_order: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_from_disk(self, filepath: str) -> None:
        """
        Parse a KIDA-format ``reactions.dat`` file and populate
        self.species, self.species_map, and self.reactions.

        Parameters
        ----------
        filepath : str
            Path to the reactions file.

        Raises
        ------
        FileNotFoundError
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Reactions file not found: '{filepath}'")
        print(f"Reading: {filepath}")
        with open(filepath) as fh:
            self._parse_lines(fh.readlines())

    def get_operators(self, env: dict):
        """
        Build the sparse ODE tensors ``A`` and ``B`` for the current
        environment.

        Parameters
        ----------
        env : dict
            Required keys: T, nH, Av, uv_flux.
            Optional: Tcap_2body (bool, default True).

        Returns
        -------
        A : scipy.sparse.csr_matrix, shape (N, N)
        B : scipy.sparse.csr_matrix, shape (N, N*N)
        """
        T = float(env["T"])
        nH = float(env["nH"])
        Av = float(env["Av"])
        uv_flux = float(env["uv_flux"])
        Tcap_2body = bool(env.get("Tcap_2body", True))

        N = len(self.species)
        A_rows, A_cols, A_data = [], [], []
        B_rows, B_cols, B_data = [], [], []

        reactions = self._select_multirange_entries(self.reactions, T)

        for rxn in reactions:
            r_idxs = [self.species_map[r] for r in rxn["reactants"]
                      if r in self.species_map]
            p_idxs = [self.species_map[p] for p in rxn["products"]
                      if p in self.species_map]

            order = len(r_idxs)
            if order == 0:
                continue

            k = self._calculate_rate(rxn, T, nH, Av, uv_flux, Tcap_2body)
            if k == 0.0:
                continue

            if order == 1:
                r1 = r_idxs[0]
                A_rows.append(r1); A_cols.append(r1); A_data.append(-k)
                for p in p_idxs:
                    A_rows.append(p); A_cols.append(r1); A_data.append(k)

            elif order == 2:
                r1, r2 = r_idxs
                col = r1 * N + r2
                B_rows.append(r1); B_cols.append(col); B_data.append(-k)
                B_rows.append(r2); B_cols.append(col); B_data.append(-k)
                for p in p_idxs:
                    B_rows.append(p); B_cols.append(col); B_data.append(k)

            else:
                if not self._warned_high_order:
                    warnings.warn(
                        "Reactions with >2 active reactants detected and "
                        "skipped (no third-order tensor is built).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_high_order = True

        def _build(rows, cols, data, shape):
            if not data:
                return sp.csr_matrix(shape, dtype=np.float64)
            mat = sp.coo_matrix(
                (np.asarray(data, dtype=np.float64),
                 (np.asarray(rows, dtype=np.int64),
                  np.asarray(cols, dtype=np.int64))),
                shape=shape,
            )
            mat.sum_duplicates()
            return mat.tocsr()

        A = _build(A_rows, A_cols, A_data, (N, N))
        B = _build(B_rows, B_cols, B_data, (N, N * N))
        return A, B

    def get_passive_species(self) -> list:
        """
        Return species that never appear as a reactant (pure-sink species).

        These species contribute no columns to A or B, so removing them
        before calling get_operators is safe and reduces the ODE dimension.

        Returns
        -------
        list[str]
        """
        active = {s for r in self.reactions for s in r["reactants"]}
        return [s for s in self.species if s not in active]

    def drop_passive_species(self) -> list:
        """
        Remove passive species from self.species and self.species_map in-place.

        Returns
        -------
        dropped : list[str]
        """
        passive = set(self.get_passive_species())
        if not passive:
            return []
        self.species = [s for s in self.species if s not in passive]
        self.species_map = {s: i for i, s in enumerate(self.species)}
        return sorted(passive)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_lines(self, lines: list) -> None:
        """Parse raw lines from a KIDA reactions file."""
        found_species: set = set()
        parsed_reactions: list = []
        _warned_K = False
        _warned_J = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("*"):
                continue

            padded = line.ljust(150)

            r_slots = [
                padded[0:11].strip(),
                padded[11:22].strip(),
                padded[22:33].strip(),
            ]
            reactants = [r for r in r_slots if r]
            products_field = padded[34:89].strip()
            products = products_field.split() if products_field else []

            all_species = reactants + products

            if any(self._is_mantle(s) for s in all_species):
                if not _warned_K:
                    warnings.warn(
                        "K-prefix (mantle) species detected and skipped.",
                        RuntimeWarning, stacklevel=2,
                    )
                    _warned_K = True
                continue

            if any(s and s[0] == "J" and len(s) > 1 for s in all_species):
                if not _warned_J:
                    warnings.warn(
                        "J-prefix (grain-surface) species detected and skipped.",
                        RuntimeWarning, stacklevel=2,
                    )
                    _warned_J = True
                continue

            params = padded[89:].replace("D", "E").split()
            if len(params) < 10:
                continue

            try:
                alpha = float(params[0])
                beta = float(params[1])
                gamma = float(params[2])
                itype = int(float(params[6]))
                tmin = float(params[7])
                tmax = float(params[8])
                frml = int(float(params[9]))
                rid = int(float(params[10])) if len(params) > 10 else -1
            except Exception as exc:
                warnings.warn(f"Skipping malformed line ({exc}): {line!r}",
                              RuntimeWarning, stacklevel=2)
                continue

            active_reactants = [r for r in reactants
                                 if r not in self.external_fields]
            if len(active_reactants) == 2:
                active_reactants = sorted(active_reactants)
            self.max_order = max(self.max_order, len(active_reactants))

            # Collect species; exclude pseudo-grain species when grains=False
            for s in active_reactants + products:
                if s and s not in self.external_fields:
                    if s in _PSEUDO_GRAIN_SPECIES and not self.grains:
                        continue
                    found_species.add(s)

            parsed_reactions.append(dict(
                reactants=active_reactants,
                products=products,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                itype=itype,
                tmin=tmin,
                tmax=tmax,
                frml=frml,
                id=rid,
            ))

        self.species = sorted(found_species)
        self.species_map = {s: i for i, s in enumerate(self.species)}
        self.reactions = parsed_reactions

        print(f"Loaded {len(self.reactions)} reactions, "
              f"{len(self.species)} species.")

    def _is_mantle(self, s: str) -> bool:
        """Return True if s is a mantle (K-prefix) species name."""
        if not s:
            return False
        # Exclude elemental potassium (K, K+, K-)
        if s in ("K",) or s.startswith("K+") or s.startswith("K-"):
            return False
        return s[0] == "K" and len(s) > 1

    def _select_multirange_entries(self, reactions: list, T: float) -> list:
        """
        For reactions with multiple temperature-range entries, keep only the
        entry whose [Tmin, Tmax] contains T, or the nearest one if T is
        out of range.  Single-entry reactions pass through unchanged.
        """
        groups: dict = defaultdict(list)
        for i, rxn in enumerate(reactions):
            key = (tuple(rxn["reactants"]), tuple(rxn["products"]), rxn["itype"])
            groups[key].append(i)

        selected: set = set()
        for indices in groups.values():
            if len(indices) == 1:
                selected.add(indices[0])
                continue
            in_range = [i for i in indices
                        if reactions[i]["tmin"] <= T <= reactions[i]["tmax"]]
            if in_range:
                selected.add(in_range[0])
            else:
                def _dist(i):
                    tmin, tmax = reactions[i]["tmin"], reactions[i]["tmax"]
                    return min(abs(T - tmin), abs(T - tmax))
                selected.add(min(indices, key=_dist))

        return [rxn for i, rxn in enumerate(reactions) if i in selected]

    def _calculate_rate(self, rxn: dict, T: float, nH: float,
                        Av: float, uv_flux: float,
                        Tcap_2body: bool) -> float:
        """Compute the effective scalar rate coefficient for a single reaction.

        For 2-body reactions nH is already absorbed, so that the contribution
        to dx/dt is k_eff * x_i * x_j with x in abundance-per-H units.

        Supported formula types
        -----------------------
        frml 1  — cosmic-ray ionisation / CR-induced photons
        frml 2  — external UV photodissociation
        frml 3  — Kooij: k = α (T/300)^β exp(−γ/T)
        frml 4  — ionpol1
        frml 5  — ionpol2
        frml 0, itype 0   — ion–grain recombination (active when grains=True)
        frml 10 — XH + XH → H₂ + H  (grains=True)
        frml 11 — H → XH  (grains=True)
        """
        frml = rxn["frml"]
        itype = rxn["itype"]
        a, b, g = rxn["alpha"], rxn["beta"], rxn["gamma"]
        tmin, tmax = rxn["tmin"], rxn["tmax"]

        # --- effective temperature (optionally clamped to [Tmin, Tmax]) ---
        def _Teff() -> float:
            if not Tcap_2body:
                return T
            if tmin <= -9000 and tmax >= 9000:
                return T
            return max(tmin, min(T, tmax))

        # --- frml 1: cosmic-ray ionisation / CR-induced photons ---
        if frml == 1:
            return a * zeta_cr

        # --- frml 2: external UV photoreactions ---
        if frml == 2:
            return a * uv_flux * np.exp(-g * Av)

        # --- frml 3/4/5: Kooij / ionpol ---
        if frml in (3, 4, 5):
            Teff = _Teff()
            if Teff <= 0.0:
                return 0.0
            if frml == 3:
                kT = a * (Teff / 300.0) ** b * np.exp(-g / Teff)
            elif frml == 4:
                kT = a * b * (0.62 + 0.4767 * g * np.sqrt(300.0 / Teff))
            else:  # frml == 5
                sq = np.sqrt(300.0 / Teff)
                kT = a * b * (1.0 + 0.0967 * g * sq + 28.501 * g * g / Teff)
            # 2-body gas-phase: absorb nH
            if itype in (4, 5, 6, 7, 8, 11):
                return kT * nH
            return kT

        Teff = _Teff()
        if Teff <= 0.0:
            return 0.0

        # --- frml 0: ion–grain recombination (itype 0 is the only frml=0
        #     reaction type present in a gas-phase reactions file) ---
        if frml == 0:
            if itype == 0:
                if self.grains:
                    return a * (Teff / 300.0) ** b * nH
                return 0.0
            raise ValueError(f"Unexpected frml=0 / itype={itype} in gas-phase file")

        # --- frml 10/11: pseudo-grain H₂ formation via XH ---
        if frml in (10, 11):
            if not self.grains:
                return 0.0
            grain_density = grain_gas_ratio * nH
            if frml == 10:
                # XH + XH → H₂ + H  (Teff is gas temperature)
                return 8.64e6 * np.exp(-225.0 / Teff) / grain_density * nH
            # frml == 11: H → XH  (adsorption pseudo-rate)
            return a * (Teff / 300.0) ** b * grain_density 

        raise ValueError(f"Unsupported formula type: frml={frml}")
