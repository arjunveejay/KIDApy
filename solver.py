"""
Solvers for the quadratic ODE system

    dx/dt = A x + B(x ⊗ x)

arising from gas-phase astrochemical reaction networks. Both classes exploit
the explicit sparse structure of A and B to provide an analytic Jacobian,
enabling efficient convergence of the stiff BDF integrator.

Classes
-------
QuadraticSolver
    Integrates the autonomous system (fixed A, B) at a single physical
    environment.

QuadraticSolverTracer
    Integrates the non-autonomous system A(p(t)), B(p(t)) along a tracer
    trajectory, with selectable interpolation of the physical conditions
    between hydrodynamic snapshots.
"""


__all__ = ["QuadraticSolver", "QuadraticSolverTracer"]

import os
import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
from typing import Tuple


class QuadraticSolver:
    """
    Solver for the quadratic ODE dx/dt = A x + B(x, x).

    A is (N x N) sparse and B is (N x N²) sparse, where column j*N+k of B
    encodes the coefficient for the bilinear term x[j]*x[k].

    All methods treat A and B as fixed for the duration of a single solve
    call.  To integrate over changing conditions, call solve repeatedly and
    chain the solutions.
    """

    # ------------------------------------------------------------------
    # RHS and Jacobian
    # ------------------------------------------------------------------

    def compute_rhs(self, x: np.ndarray, A: sp.spmatrix, B: sp.spmatrix) -> np.ndarray:
        """
        Evaluate dx/dt = A x + B(x, x).

        Parameters
        ----------
        x : np.ndarray, shape (N,)
        A : scipy.sparse matrix, shape (N, N)
        B : scipy.sparse matrix, shape (N, N*N)

        Returns
        -------
        dxdt : np.ndarray, shape (N,)
        """
        x = np.asarray(x, dtype=np.float64)
        dxdt = (A @ x).astype(np.float64, copy=False)

        if B.nnz == 0:
            return dxdt

        N = x.size
        Bcoo = B.tocoo(copy=False)
        j = (Bcoo.col // N).astype(np.int64, copy=False)
        k = (Bcoo.col  % N).astype(np.int64, copy=False)
        np.add.at(dxdt, Bcoo.row, Bcoo.data * x[j] * x[k])
        return dxdt

    def compute_jacobian(self, x: np.ndarray, A: sp.spmatrix, B: sp.spmatrix) -> sp.csr_matrix:
        """
        Evaluate the Jacobian J = A + B(x, ·) + B(·, x).

        Parameters
        ----------
        x : np.ndarray, shape (N,)
        A : scipy.sparse matrix, shape (N, N)
        B : scipy.sparse matrix, shape (N, N*N)

        Returns
        -------
        J : scipy.sparse.csr_matrix, shape (N, N)
        """
        x = np.asarray(x, dtype=np.float64)
        N = x.size
        J = A.tocsr(copy=True)

        if B.nnz == 0:
            return J

        Bcoo = B.tocoo(copy=False)
        i   = Bcoo.row.astype(np.int64, copy=False)
        col = Bcoo.col.astype(np.int64, copy=False)
        j   = (col // N).astype(np.int64, copy=False)
        k   = (col  % N).astype(np.int64, copy=False)
        v   = np.asarray(Bcoo.data, dtype=np.float64)

        Jq = sp.coo_matrix(
            (np.concatenate([v * x[k], v * x[j]]),
             (np.concatenate([i, i]), np.concatenate([j, k]))),
            shape=(N, N), dtype=np.float64,
        )
        Jq.sum_duplicates()
        return (J + Jq.tocsr()).tocsr()

    # ------------------------------------------------------------------
    # Solve function
    # ------------------------------------------------------------------

    def solve(
        self,
        A: sp.spmatrix,
        B: sp.spmatrix,
        t_span: Tuple[float, float],
        x0: np.ndarray,
        atol: float,
        rtol: float,
        method: str = "BDF",
        t_eval=None,
        scale: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate dx/dt = A x + B(x, x) over [t_span[0], t_span[1]].

        Parameters
        ----------
        A : scipy.sparse matrix, shape (N, N)
        B : scipy.sparse matrix, shape (N, N*N)
        t_span : (t0, t1)
        x0 : np.ndarray, shape (N,)
            Initial state.
        atol : float
            Absolute tolerance.  When scale is provided this applies to the
            O(1) scaled variables; otherwise it applies directly to x.
        rtol : float
            Relative tolerance.
        method : str
            Integration method passed to scipy.integrate.solve_ivp.
            Default: "BDF".
        t_eval : array-like or None
            Times at which to store the solution.
        scale : np.ndarray, shape (N,), or None
            When provided the system is solved in scaled coordinates
            z = x / scale, then unscaled before returning.  scale must be
            positive.  Useful when x spans many orders of magnitude.

        Returns
        -------
        t : np.ndarray, shape (Nt,)
        y : np.ndarray, shape (N, Nt)

        Raises
        ------
        RuntimeError
            If the solver reports a failure (sol.status != 0).
        """
        N  = len(x0)
        x0 = np.asarray(x0, dtype=np.float64)
        Acsr = A.tocsr(copy=False)

        if scale is not None:
            s = np.asarray(scale, dtype=np.float64)
            if s.shape != (N,) or np.any(s <= 0):
                raise ValueError("scale must be a positive array of shape (N,)")
            s_inv = 1.0 / s
            z0    = x0 * s_inv

            A_sc = sp.diags(s_inv, format="csr") @ Acsr @ sp.diags(s, format="csr")

            if B.nnz > 0:
                Bcoo = B.tocoo(copy=False)
                bi = Bcoo.row.astype(np.int64, copy=False)
                bj = (Bcoo.col // N).astype(np.int64, copy=False)
                bk = (Bcoo.col  % N).astype(np.int64, copy=False)
                B_sc = sp.coo_matrix(
                    (Bcoo.data * s[bj] * s[bk] * s_inv[bi], (Bcoo.row, Bcoo.col)),
                    shape=B.shape, dtype=np.float64,
                ).tocsr()

                Bcoo_sc = B_sc.tocoo(copy=False)
                bi_sc = Bcoo_sc.row.astype(np.int64, copy=False)
                bj_sc = (Bcoo_sc.col // N).astype(np.int64, copy=False)
                bk_sc = (Bcoo_sc.col  % N).astype(np.int64, copy=False)
                bv_sc = np.asarray(Bcoo_sc.data, dtype=np.float64)
                jrows = np.concatenate([bi_sc, bi_sc])
                jcols = np.concatenate([bj_sc, bk_sc])

                def jac(_t, z):
                    data = np.concatenate([bv_sc * z[bk_sc], bv_sc * z[bj_sc]])
                    Jq = sp.coo_matrix((data, (jrows, jcols)), shape=(N, N), dtype=np.float64)
                    Jq.sum_duplicates()
                    return (A_sc + Jq.tocsr()).tocsr()
            else:
                B_sc = B

                def jac(_t, _z):
                    return A_sc

            def f(_t, z):
                return self.compute_rhs(z, A_sc, B_sc)

            sol = solve_ivp(f, t_span, z0, method=method, atol=atol,
                            rtol=rtol, jac=jac, t_eval=t_eval)
            if sol.status == 0:
                return sol.t, s[:, None] * sol.y
            raise RuntimeError(f"Solver failed (status={sol.status}): {sol.message}")

        # --- unscaled ---
        if B.nnz > 0:
            Bcoo = B.tocoo(copy=False)
            bi   = Bcoo.row.astype(np.int64, copy=False)
            bj   = (Bcoo.col // N).astype(np.int64, copy=False)
            bk   = (Bcoo.col  % N).astype(np.int64, copy=False)
            bv   = np.asarray(Bcoo.data, dtype=np.float64)
            jrows = np.concatenate([bi, bi])
            jcols = np.concatenate([bj, bk])

            def jac(t, x):
                data = np.concatenate([bv * x[bk], bv * x[bj]])
                Jq = sp.coo_matrix((data, (jrows, jcols)), shape=(N, N), dtype=np.float64)
                Jq.sum_duplicates()
                return (Acsr + Jq.tocsr()).tocsr()
        else:
            def jac(t, x):
                return Acsr

        def f(t, x):
            return self.compute_rhs(x, A, B)

        sol = solve_ivp(f, t_span, x0, method=method, atol=atol,
                        rtol=rtol, jac=jac, t_eval=t_eval)
        if sol.status == 0:
            return sol.t, sol.y
        raise RuntimeError(f"Solver failed (status={sol.status}): {sol.message}")

    # ------------------------------------------------------------------
    # Save and plot
    # ------------------------------------------------------------------

    def save(
        self,
        path: str,
        t: np.ndarray,
        y: np.ndarray,
        col_names=None,
    ):
        """
        Save a solution to .npy and .csv files.

        The output array has shape (Nt, 1 + N): the first column is t,
        followed by one column per state variable.

        Parameters
        ----------
        path : str
            Base path.  Any extension is stripped; .npy and .csv are written
            at {root}.npy and {root}.csv.
        t : np.ndarray, shape (Nt,)
        y : np.ndarray, shape (N, Nt)
        col_names : list of str or None
            Names for the N state-variable columns.  Used as the CSV header.
            If None, columns are named x0, x1, ..., x{N-1}.

        Returns
        -------
        out : np.ndarray, shape (Nt, 1 + N)
        header : list of str
        """
        t = np.asarray(t, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.float64)
        if y.shape[1] != t.size:
            raise ValueError(f"y.shape[1]={y.shape[1]} must equal len(t)={t.size}")

        N = y.shape[0]
        if col_names is None:
            col_names = [f"x{i}" for i in range(N)]
        if len(col_names) != N:
            raise ValueError(f"len(col_names)={len(col_names)} must equal N={N}")

        out = np.vstack([t, y]).T
        header = ["t"] + list(col_names)

        root = os.path.splitext(path)[0] or path
        np.save(root + ".npy", out)
        np.savetxt(root + ".csv", out, delimiter=",",
                   header=",".join(header), comments="")
        return out, header

    def plot(
        self,
        t: np.ndarray,
        y: np.ndarray,
        indices,
        labels=None,
        outpath: str = None,
    ):
        """
        Plot selected rows of y against t on a log-log scale.

        Parameters
        ----------
        t : np.ndarray, shape (Nt,)
        y : np.ndarray, shape (N, Nt)
        indices : array-like of int
            Row indices of y to plot.
        labels : list of str or None
            Legend labels, one per index.  If None, labels are the indices.
        outpath : str or None
            If provided, save the figure to this path.

        Returns
        -------
        fig : matplotlib.figure.Figure
        ax : matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt

        indices = list(indices)
        if labels is None:
            labels = [str(i) for i in indices]
        if len(labels) != len(indices):
            raise ValueError("labels must have the same length as indices")

        cmap = plt.get_cmap("tab20", len(indices))
        fig, ax = plt.subplots(figsize=(6, 4))

        for i, (idx, label) in enumerate(zip(indices, labels)):
            ax.plot(t, y[idx], "-", color=cmap(i), label=label)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.grid(True)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.subplots_adjust(right=0.78)

        if outpath is not None:
            fig.savefig(outpath, bbox_inches="tight")

        return fig, ax


class QuadraticSolverTracer:
    """
    Solver for chemistry along a tracer trajectory with selectable
    interpolation of the physical conditions.

    The solver advances the non-autonomous quadratic system

        dx/dt = A(p(t)) x + B(p(t))(x, x)

    where ``p(t) = [nH(t), T(t), Tgrain(t), Av(t), uv_flux(t)]`` is the
    tracer's hydrodynamic trajectory. For ``"piecewise_constant"``,
    ``A(p(t))`` and ``B(p(t))`` are held fixed over each hydro step. For
    ``"cubic_spline"`` and ``"pchip"``, ``p(t)`` is interpolated
    continuously in time before calling ``get_tensors``.

    Supported interpolation modes are:
    ``"piecewise_constant"``, ``"cubic_spline"``, and ``"pchip"``.
    """

    _PT_COLS = ("nH", "T", "Tgrain", "Av", "uv_flux")
    _VALID_INTERPOLATION = {"piecewise_constant", "cubic_spline", "pchip"}

    def solve(
        self,
        dt_hydro: float,
        pt: np.ndarray,
        get_tensors,
        x0: np.ndarray,
        atol: float,
        rtol: float,
        min_scale: float,
        method: str = "BDF",
        interpolation: str = "piecewise_constant",
        use_scaling: bool = False,
        verbose: bool = False,
        log_cols=None,
        t_eval=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the time-dependent system

            dx/dt = A(p(t)) x + B(p(t))(x, x)

        along a tracer trajectory.

        Parameters
        ----------
        dt_hydro : float
            Hydrodynamic timestep in seconds.
        pt : np.ndarray, shape (M, 5)
            Physical trajectory with columns ``[nH, T, Tgrain, Av, uv_flux]``.
        get_tensors : callable
            Function ``get_tensors(env) -> (A, B)``.
        x0 : np.ndarray, shape (N,)
            Initial species abundances.
        atol : float
            Absolute tolerance for the hydro-step ODE solves.
        rtol : float
            Relative tolerance for all ODE solves.
        min_scale : float
            Lower bound used when constructing the scale vector for scaled
            solves. It is also used in the tracer's internal floor logic.
        method : str, optional
            ODE integration method passed to ``scipy.integrate.solve_ivp``.
        interpolation : {"piecewise_constant", "cubic_spline", "pchip"}, optional
            Interpolation used for the physical trajectory.
        use_scaling : bool, optional
            If ``True``, solve each segment in scaled coordinates.
        verbose : bool, optional
            If ``True``, print per-segment solver statistics.
        log_cols : iterable of str or None, optional
            Columns to interpolate in log space for spline or PCHIP modes.
        t_eval : array-like or None, optional
            Optional output time grid over the full trajectory.

        Returns
        -------
        t : np.ndarray, shape (Nt,)
            Output time points.
        y : np.ndarray, shape (N, Nt)
            Species abundances at the output times.
        """
        interpolation = str(interpolation)
        if interpolation not in self._VALID_INTERPOLATION:
            raise ValueError(
                f"interpolation must be one of {sorted(self._VALID_INTERPOLATION)}; "
                f"got {interpolation!r}"
            )

        pt = np.asarray(pt, dtype=np.float64)
        if pt.ndim != 2 or pt.shape[1] != len(self._PT_COLS):
            raise ValueError(
                f"pt must have shape (M, {len(self._PT_COLS)}); got {pt.shape}"
            )
        if pt.shape[0] == 0:
            raise ValueError("pt must contain at least one trajectory point")

        x0 = np.asarray(x0, dtype=np.float64).copy()
        x_floor = min_scale * atol if use_scaling else atol
        q = QuadraticSolver()

        A0, B0 = get_tensors(self._to_env(pt[0]))
        _, y_eq = q.solve(
            A0,
            B0,
            (0.0, 3600 * 24 * 365 * 1e4),
            x0,
            method=method,
            atol=x_floor,
            rtol=rtol,
            scale=None,
        )
        x0 = y_eq[:, -1]

        M = pt.shape[0]
        if M == 1:
            return np.array([0.0], dtype=np.float64), x0[:, None]

        if interpolation == "piecewise_constant":
            return self._solve_piecewise_constant(
                q=q,
                dt_hydro=dt_hydro,
                pt=pt,
                get_tensors=get_tensors,
                x0=x0,
                atol=atol,
                rtol=rtol,
                min_scale=min_scale,
                method=method,
                use_scaling=use_scaling,
                verbose=verbose,
                t_eval=t_eval,
            )

        N = x0.size
        t_knots = np.arange(M, dtype=np.float64) * dt_hydro
        eval_env = self._build_env_evaluator(
            t_knots=t_knots,
            pt=pt,
            interpolation=interpolation,
            log_cols=log_cols,
        )

        cache = [None, None, None]

        def _AB(t: float):
            t_c = float(np.clip(t, t_knots[0], t_knots[-1]))
            if cache[0] == t_c:
                return cache[1], cache[2]
            env_vals = eval_env(t_c)
            cache[1], cache[2] = get_tensors(self._to_env(env_vals))
            cache[0] = t_c
            return cache[1], cache[2]

        t_eval_arr = np.asarray(t_eval, dtype=np.float64) if t_eval is not None else None
        all_t, all_y = [], []

        for i in range(M - 1):
            t_span_i = (t_knots[i], t_knots[i + 1])

            if t_eval_arr is not None:
                upper_cmp = (t_eval_arr <= t_span_i[1]) if i == M - 2 else (t_eval_arr < t_span_i[1])
                seg_t_eval = t_eval_arr[(t_eval_arr >= t_span_i[0]) & upper_cmp]
                seg_t_eval = seg_t_eval if seg_t_eval.size else None
            else:
                seg_t_eval = None

            if use_scaling:
                scale = np.maximum(x0, min_scale)
                s_inv = 1.0 / scale

                def _scale_AB(A_t, B_t, scale=scale, s_inv=s_inv):
                    A_sc = sp.diags(s_inv, format="csr") @ A_t.tocsr() @ sp.diags(scale, format="csr")
                    if B_t.nnz == 0:
                        return A_sc, B_t
                    Bcoo = B_t.tocoo(copy=False)
                    bi = Bcoo.row.astype(np.int64, copy=False)
                    bj = (Bcoo.col // N).astype(np.int64, copy=False)
                    bk = (Bcoo.col % N).astype(np.int64, copy=False)
                    B_sc = sp.coo_matrix(
                        (Bcoo.data * scale[bj] * scale[bk] * s_inv[bi], (Bcoo.row, Bcoo.col)),
                        shape=B_t.shape,
                        dtype=np.float64,
                    ).tocsr()
                    return A_sc, B_sc

                def f(t, z, _scale_AB=_scale_AB):
                    return q.compute_rhs(z, *_scale_AB(*_AB(t)))

                def jac(t, z, _scale_AB=_scale_AB):
                    return q.compute_jacobian(z, *_scale_AB(*_AB(t)))

                z0 = x0 * s_inv
                sol = solve_ivp(
                    f,
                    t_span_i,
                    z0,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    jac=jac,
                    t_eval=seg_t_eval,
                )
                if sol.status != 0:
                    raise RuntimeError(
                        f"Solver failure at segment {i} with status={sol.status}: {sol.message}"
                    )
                y_seg = scale[:, None] * sol.y
            else:
                def f(t, x):
                    A_t, B_t = _AB(t)
                    return q.compute_rhs(x, A_t, B_t)

                def jac(t, x):
                    A_t, B_t = _AB(t)
                    return q.compute_jacobian(x, A_t, B_t)

                sol = solve_ivp(
                    f,
                    t_span_i,
                    x0,
                    method=method,
                    atol=atol,
                    rtol=rtol,
                    jac=jac,
                    t_eval=seg_t_eval,
                )
                if sol.status != 0:
                    raise RuntimeError(
                        f"Solver failure at segment {i} with status={sol.status}: {sol.message}"
                    )
                y_seg = sol.y

            all_t.append(sol.t)
            all_y.append(y_seg)
            x0 = y_seg[:, -1]

            if verbose:
                print(f"{interpolation} segment {i + 1}/{M - 1}: nfev={sol.nfev}  njev={sol.njev}  nout={sol.t.size}")

        t_out = [all_t[0]]
        y_out = [all_y[0]]
        for i in range(1, len(all_t)):
            t_out.append(all_t[i][1:])
            y_out.append(all_y[i][:, 1:])
        return np.concatenate(t_out), np.hstack(y_out)

    def save_data(
        self,
        t,
        y,
        pt: np.ndarray,
        species: list,
        save_path: str = "run",
        save_csv: bool = True,
        dt_hydro: float = None,
        interpolation: str = "piecewise_constant",
        log_cols=None,
    ):
        """
        Save tracer output to ``.npy`` and optionally ``.csv``.

        For ``interpolation="piecewise_constant"``, ``t`` and ``y`` should be
        the ``(all_t, all_y)`` lists returned by :meth:`solve`. For
        ``"cubic_spline"`` and ``"pchip"``, ``t`` and ``y`` should be the
        continuous ``(t, y)`` arrays returned by :meth:`solve`, and
        ``dt_hydro`` must be provided so the physical trajectory can be
        reconstructed at each output time.
        """
        interpolation = str(interpolation)
        if interpolation not in self._VALID_INTERPOLATION:
            raise ValueError(
                f"interpolation must be one of {sorted(self._VALID_INTERPOLATION)}; "
                f"got {interpolation!r}"
            )

        pt = np.asarray(pt, dtype=np.float64)
        if interpolation == "piecewise_constant":
            return self._save_piecewise_constant_data(t, y, pt, species, save_path, save_csv)

        if dt_hydro is None:
            raise ValueError("dt_hydro is required when saving spline or PCHIP tracer output.")

        t = np.asarray(t, dtype=np.float64).ravel()
        Nt = t.size
        sol = np.asarray(y, dtype=np.float64)
        if sol.ndim != 2:
            raise ValueError(f"y must be 2D; got {sol.shape}")
        if sol.shape[1] != Nt:
            sol = sol.T
        if sol.shape[1] != Nt:
            raise ValueError(f"y shape {sol.shape} incompatible with t length {Nt}")

        t_knots = np.arange(pt.shape[0], dtype=np.float64) * dt_hydro
        eval_env = self._build_env_evaluator(
            t_knots=t_knots,
            pt=pt,
            interpolation=interpolation,
            log_cols=log_cols,
        )

        p_keys = ["T", "nH", "Av", "uv_flux", "Tgrain"]
        header = p_keys + ["t"] + list(species) + ["IC"]
        header_line = ",".join(header)

        p_vals = np.array([[self._to_env(eval_env(ti))[k] for k in p_keys] for ti in t], dtype=np.float64)
        IC = np.zeros((Nt, 1), dtype=np.float64)
        IC[0, 0] = float(Nt)
        out = np.hstack((p_vals, np.vstack((t, sol)).T, IC))

        root, ext = os.path.splitext(save_path)
        root = root if ext else save_path
        np.save(root + ".npy", out)
        if save_csv:
            np.savetxt(root + ".csv", out, delimiter=",", header=header_line, comments="")
        return out, header

    def _save_piecewise_constant_data(
        self,
        all_t,
        all_y,
        pt: np.ndarray,
        species: list,
        save_path: str,
        save_csv: bool,
    ):
        p_keys = ["T", "nH", "Av", "uv_flux", "Tgrain"]
        header = p_keys + ["t"] + list(species) + ["IC"]
        header_line = ",".join(header)

        chunks = []
        for i, (t_i, y_i) in enumerate(zip(all_t, all_y)):
            p = self._to_env(pt[i])
            t_i = np.asarray(t_i, dtype=np.float64).ravel()
            N_t = t_i.size

            sol = np.asarray(y_i, dtype=np.float64)
            if sol.ndim != 2:
                raise ValueError(f"sol must be 2D; got {sol.shape}")
            if sol.shape[1] != N_t:
                sol = sol.T
            if sol.shape[1] != N_t:
                raise ValueError(f"sol shape {sol.shape} incompatible with t length {N_t}")

            if i > 0:
                t_i = t_i[1:]
                sol = sol[:, 1:]
                N_t = t_i.size
                if N_t == 0:
                    continue

            block = np.vstack((t_i, sol)).T
            p_vals = np.array([p[k] for k in p_keys], dtype=np.float64)
            p_cols = np.tile(p_vals, (N_t, 1))
            IC = np.zeros((N_t, 1), dtype=np.float64)
            IC[0, 0] = float(N_t)
            chunks.append(np.hstack((p_cols, block, IC)))

        all_out = np.vstack(chunks)
        root, ext = os.path.splitext(save_path)
        root = root if ext else save_path
        np.save(root + ".npy", all_out)
        if save_csv:
            np.savetxt(root + ".csv", all_out, delimiter=",", header=header_line, comments="")
        return all_out, header

    def _solve_piecewise_constant(
        self,
        q: "QuadraticSolver",
        dt_hydro: float,
        pt: np.ndarray,
        get_tensors,
        x0: np.ndarray,
        atol: float,
        rtol: float,
        min_scale: float,
        method: str,
        use_scaling: bool,
        verbose: bool,
        t_eval=None,
    ):
        #x_floor = min_scale * atol if use_scaling else atol
        t0 = 0.0
        t1 = float(dt_hydro)
        M = pt.shape[0]
        t_knots = np.arange(M, dtype=np.float64) * dt_hydro
        t_eval_arr = np.asarray(t_eval, dtype=np.float64) if t_eval is not None else None

        if use_scaling:
            scale = np.maximum(x0, min_scale)
            if verbose:
                print(f"scale  range : [{scale.min():.3e}, {scale.max():.3e}]")
        else:
            scale = None

        all_t, all_y = [], []
        for i in range(M - 1):
            A_i, B_i = get_tensors(self._to_env(pt[i]))
            if t_eval_arr is not None:
                t_span_i = (t_knots[i], t_knots[i + 1])
                seg_t_eval_abs = t_eval_arr[(t_eval_arr >= t_span_i[0]) & (t_eval_arr <= t_span_i[1])]
                # For piecewise plotting/output we want both exact segment
                # endpoints present, even if the user-supplied global grid
                # does not land on the knots.
                seg_t_eval_abs = np.unique(
                    np.concatenate(
                        [
                            np.array([t_span_i[0]], dtype=np.float64),
                            seg_t_eval_abs,
                            np.array([t_span_i[1]], dtype=np.float64),
                        ]
                    )
                )
                seg_t_eval = seg_t_eval_abs - i * dt_hydro
                seg_t_eval = np.clip(seg_t_eval, t0, t1)
            else:
                seg_t_eval = None
            t_i, y_i = q.solve(
                A_i,
                B_i,
                (t0, t1),
                x0,
                method=method,
                atol=atol,
                rtol=rtol,
                scale=scale,
                t_eval=seg_t_eval,
            )

            all_t.append(t_i + i * dt_hydro)
            all_y.append(y_i)
            x0 = y_i[:, -1]

            if verbose:
                step_str = f"step {i + 1:>{len(str(M))}}/{M}"
                if use_scaling:
                    z_end = x0 / scale
                    print(
                        f"{step_str}  "
                        f"z: min={z_end.min():.3e}  max={z_end.max():.3e}  mean={z_end.mean():.3e}"
                    )
                else:
                    print(
                        f"{step_str}  "
                        f"x: min={x0.min():.3e}  max={x0.max():.3e}  mean={x0.mean():.3e}"
                    )
            if use_scaling:
                scale = np.maximum(x0, min_scale)

        return all_t, all_y

    def _build_env_evaluator(
        self,
        t_knots: np.ndarray,
        pt: np.ndarray,
        interpolation: str,
        log_cols=None,
    ):
        if interpolation == "piecewise_constant":
            def eval_env(t: float) -> np.ndarray:
                idx = np.searchsorted(t_knots, t, side="left") - 1
                idx = int(np.clip(idx, 0, pt.shape[0] - 1))
                return pt[idx]

            return eval_env

        log_mask = np.zeros(pt.shape[1], dtype=bool)
        if log_cols is not None:
            name_to_idx = {name: i for i, name in enumerate(self._PT_COLS)}
            unknown = [name for name in log_cols if name not in name_to_idx]
            if unknown:
                raise ValueError(
                    f"log_cols contains unknown parameter name(s) {unknown}; "
                    f"valid names are {list(self._PT_COLS)}."
                )
            log_mask[[name_to_idx[name] for name in log_cols]] = True
            if not (pt[:, log_mask] > 0).all():
                raise ValueError("log_cols must refer to columns of pt with all-positive values.")

        pt_fit = pt.astype(np.float64, copy=True)
        pt_fit[:, log_mask] = np.log(pt_fit[:, log_mask])

        if interpolation == "cubic_spline":
            interpolator = CubicSpline(t_knots, pt_fit, axis=0, extrapolate=False)
        else:
            interpolator = PchipInterpolator(t_knots, pt_fit, axis=0, extrapolate=False)

        def eval_env(t: float) -> np.ndarray:
            t_c = float(np.clip(t, t_knots[0], t_knots[-1]))
            vals = np.asarray(interpolator(t_c), dtype=np.float64)
            if log_mask.any():
                vals = vals.copy()
                vals[log_mask] = np.exp(vals[log_mask])
            return vals

        return eval_env

    def _to_env(self, p: np.ndarray) -> dict:
        return dict(
            T=float(p[1]),
            nH=float(p[0]),
            Av=float(p[3]),
            uv_flux=float(p[4]),
            Tcap_2body=True,
            Tgrain=float(p[2]),
        )
