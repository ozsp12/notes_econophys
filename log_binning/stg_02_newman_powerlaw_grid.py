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
ASSET_DIR = ROOT / "assets" / "newman_powerlaw"

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


def _prepare_panel_data(values: np.ndarray, alpha: float, xmin: float) -> dict[str, np.ndarray]:
    """Prepare all numerical arrays used by the four diagnostic panels."""
    n = len(values)

    x_linear_max = 8.0 * xmin
    x_pdf = np.linspace(xmin, x_linear_max, 800)
    normalization = (alpha - 1.0) * xmin ** (alpha - 1.0)
    pdf = normalization * x_pdf ** (-alpha)

    linear_hist_max = min(float(np.quantile(values, 0.9995)), 200.0 * xmin)
    linear_edges = np.linspace(xmin, linear_hist_max, 201)
    linear_counts, linear_edges = np.histogram(values, bins=linear_edges)
    linear_centers = 0.5 * (linear_edges[:-1] + linear_edges[1:])
    linear_frequencies = linear_counts / n

    log_hist_max = float(np.quantile(values, 0.99999))
    log_edges = np.geomspace(xmin, log_hist_max, 90)
    log_counts, log_edges = np.histogram(values, bins=log_edges)
    log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])
    log_widths = np.diff(log_edges)
    log_density = log_counts / (n * log_widths)

    ccdf_x, ccdf = empirical_ccdf(values)

    return {
        "x_linear_max": np.array([x_linear_max]),
        "normalization": np.array([normalization]),
        "x_pdf": x_pdf,
        "pdf": pdf,
        "linear_counts": linear_counts,
        "linear_centers": linear_centers,
        "linear_frequencies": linear_frequencies,
        "log_counts": log_counts,
        "log_centers": log_centers,
        "log_density": log_density,
        "ccdf_x": ccdf_x,
        "ccdf": ccdf,
    }


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="major", length=8, width=1.0)
    ax.tick_params(which="minor", length=4, width=0.8)


def _draw_panel_a(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    ax.plot(data["x_pdf"], data["pdf"], linewidth=1.2)
    ax.set_xlim(0.0, float(data["x_linear_max"][0]))
    ax.set_ylim(0.0, 1.5 * float(data["normalization"][0]))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("samples")
    ax.text(0.56, 0.88, "(a)", transform=ax.transAxes, fontsize=15)
    _style_axis(ax)


def _draw_panel_b(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    mask = data["linear_counts"] > 0
    ax.plot(
        data["linear_centers"][mask],
        data["linear_frequencies"][mask],
        linewidth=1.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("samples")
    ax.text(0.56, 0.88, "(b)", transform=ax.transAxes, fontsize=15)
    _style_axis(ax)


def _draw_panel_c(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    mask = data["log_counts"] > 0
    ax.plot(
        data["log_centers"][mask],
        data["log_density"][mask],
        linewidth=1.0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("samples")
    ax.text(0.56, 0.88, "(c)", transform=ax.transAxes, fontsize=15)
    _style_axis(ax)


def _draw_panel_d(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    ax.plot(data["ccdf_x"], data["ccdf"], linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"samples with value $>x$")
    ax.text(0.56, 0.88, "(d)", transform=ax.transAxes, fontsize=15)
    _style_axis(ax)


PANEL_DRAWERS = (_draw_panel_a, _draw_panel_b, _draw_panel_c, _draw_panel_d)


def _save_individual_panels(data: dict[str, np.ndarray], asset_dir: Path) -> None:
    """Save the four component panels as separate reproducible assets."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    for label, drawer in zip("abcd", PANEL_DRAWERS):
        fig, ax = plt.subplots(figsize=(5.1, 4.0))
        drawer(ax, data)
        fig.tight_layout()
        fig.savefig(asset_dir / f"newman_panel_{label}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def make_newman_powerlaw_grid(
    values: np.ndarray | None = None,
    *,
    alpha: float = ALPHA,
    xmin: float = XMIN,
    output_path: str | Path = FIGURE_PATH,
    asset_dir: str | Path = ASSET_DIR,
    show: bool = False,
) -> Path:
    """Create a 2x2 power-law diagnostic grid inspired by Newman.

    In addition to the complete grid, the four component panels are saved as
    individual PNG assets so that the note can reuse or rearrange them without
    regenerating the sample.
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
    asset_dir = Path(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    data = _prepare_panel_data(values, alpha, xmin)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes
    for ax, drawer in zip((ax_a, ax_b, ax_c, ax_d), PANEL_DRAWERS):
        drawer(ax, data)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.98,
        wspace=0.36,
        hspace=0.34,
    )

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(asset_dir / "newman_powerlaw_grid.png", dpi=200, bbox_inches="tight")
    _save_individual_panels(data, asset_dir)

    if show:
        plt.show()
    plt.close(fig)
    return output_path


def main() -> None:
    path = make_newman_powerlaw_grid()
    print(f"Generated {path.name} and assets in {ASSET_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
