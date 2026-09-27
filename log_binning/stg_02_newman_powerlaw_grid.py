"""Generate a Newman-style 2x2 diagnostic grid for a Pareto power law.

The four panels compare complementary representations of the same synthetic
power-law sample: a linear-linear theoretical density, a conventional histogram
shown on log-log axes, a logarithmically binned density estimate, and the
empirical complementary cumulative distribution function (CCDF).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FIGURE_PATH = ROOT / "newman_powerlaw_grid.png"

SEED = 20260926
N = 1_000_000
ALPHA = 2.5
XMIN = 1.0


def generate_pareto_sample(
    n: int = N,
    alpha: float = ALPHA,
    xmin: float = XMIN,
    seed: int = SEED,
) -> np.ndarray:
    """Generate a continuous Pareto sample with p(x) proportional to x^-alpha."""
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1.")
    if xmin <= 0.0:
        raise ValueError("xmin must be strictly positive.")

    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def empirical_ccdf(
    values: np.ndarray,
    max_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a tail-resolved empirical CCDF with logarithmic down-sampling."""
    x = np.sort(values)
    n = len(x)
    remaining = np.unique(
        np.rint(np.geomspace(1, n, max_points)).astype(np.int64)
    )
    idx = np.sort(n - remaining)
    return x[idx], (n - idx) / n


def make_newman_powerlaw_grid(
    values: np.ndarray | None = None,
    *,
    alpha: float = ALPHA,
    xmin: float = XMIN,
    output_path: str | Path = FIGURE_PATH,
    show: bool = False,
) -> Path:
    """Create a 2x2 power-law diagnostic grid inspired by Newman.

    Panels
    ------
    (a) Theoretical Pareto probability density on linear axes.
    (b) Equal-width histogram frequencies on log-log axes.
    (c) Logarithmically binned density estimate on log-log axes.
    (d) Empirical CCDF on log-log axes.

    Parameters
    ----------
    values:
        Optional positive sample. If omitted, a deterministic Pareto sample is
        generated from ``alpha`` and ``xmin``.
    alpha:
        Pareto PDF exponent, p(x) proportional to x^-alpha.
    xmin:
        Lower cutoff of the Pareto distribution.
    output_path:
        Destination PNG path.
    show:
        Display the figure interactively when True.
    """
    if values is None:
        values = generate_pareto_sample(alpha=alpha, xmin=xmin)
    else:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("values must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("values must contain only finite positive numbers.")

    output_path = Path(output_path)
    n = len(values)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) Theoretical density in linear-linear coordinates.
    x_linear_max = 8.0 * xmin
    x_pdf = np.linspace(xmin, x_linear_max, 800)
    normalization = (alpha - 1.0) * xmin ** (alpha - 1.0)
    pdf = normalization * x_pdf ** (-alpha)

    ax_a.plot(x_pdf, pdf, linewidth=1.2)
    ax_a.set_xlim(0.0, x_linear_max)
    ax_a.set_ylim(0.0, 1.5 * normalization)
    ax_a.set_xlabel(r"$x$")
    ax_a.set_ylabel("samples")
    ax_a.text(0.56, 0.88, "(a)", transform=ax_a.transAxes, fontsize=15)

    # (b) Equal-width histogram frequencies on log-log axes.
    linear_hist_max = min(float(np.quantile(values, 0.9995)), 200.0 * xmin)
    linear_edges = np.linspace(xmin, linear_hist_max, 201)
    counts, edges = np.histogram(values, bins=linear_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    frequencies = counts / n
    mask = counts > 0

    ax_b.plot(centers[mask], frequencies[mask], linewidth=1.0)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlabel(r"$x$")
    ax_b.set_ylabel("samples")
    ax_b.text(0.56, 0.88, "(b)", transform=ax_b.transAxes, fontsize=15)

    # (c) Logarithmic-bin density estimate on log-log axes.
    log_hist_max = float(np.quantile(values, 0.99999))
    log_edges = np.geomspace(xmin, log_hist_max, 90)
    counts, edges = np.histogram(values, bins=log_edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    density = counts / (n * widths)
    mask = counts > 0

    ax_c.plot(centers[mask], density[mask], linewidth=1.0)
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(r"$x$")
    ax_c.set_ylabel("samples")
    ax_c.text(0.56, 0.88, "(c)", transform=ax_c.transAxes, fontsize=15)

    # (d) Empirical complementary cumulative distribution function.
    ccdf_x, ccdf = empirical_ccdf(values)
    ax_d.plot(ccdf_x, ccdf, linewidth=1.0)
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlabel(r"$x$")
    ax_d.set_ylabel(r"samples with value $>x$")
    ax_d.text(0.56, 0.88, "(d)", transform=ax_d.transAxes, fontsize=15)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(which="major", length=8, width=1.0)
        ax.tick_params(which="minor", length=4, width=0.8)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    path = make_newman_powerlaw_grid()
    print(f"Generated {path.name}")


if __name__ == "__main__":
    main()
