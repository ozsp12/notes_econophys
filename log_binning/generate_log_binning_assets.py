"""Generate all numerical tables and figures for the log-binning notes.

All generated outputs are stored directly in ``log_binning``. Figure files use
``figure_`` as a prefix and tabular data files use ``table_`` as a prefix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLE_PATH = ROOT / "table_binning_experiments.csv"

FIGURE_HISTOGRAMS = ROOT / "figure_histograms.png"
FIGURE_CCDF = ROOT / "figure_ccdf_grid.png"
FIGURE_CCDF_NO_TITLE = ROOT / "figure_ccdf_grid_no_title.png"
FIGURE_BIN_MEANS = ROOT / "figure_bin_means_grid.png"
FIGURE_NEWMAN_GRID = ROOT / "figure_newman_powerlaw_grid.png"

SEED = 20260925
NEWMAN_SEED = 20260926
N = 1_000_000
HISTOGRAM_N = 100_000
K_VALUES = (25, 50, 100)
DISTRIBUTIONS = ("exponential", "lognormal", "pareto")
METHODS = ("cut", "qcut", "log_binning")

EXPONENTIAL_SCALE = 50.0
LOGNORMAL_MU = float(np.log(150.0))
LOGNORMAL_SIGMA = 0.55
PARETO_ALPHA_PDF = 2.5
PARETO_XMIN = 100.0
PARETO_DISPLAY_QUANTILE = 0.99995

NEWMAN_ALPHA = 2.5
NEWMAN_XMIN = 1.0

HISTOGRAM_COLORS = {
    "exponential": "tab:blue",
    "lognormal": "tab:orange",
    "pareto": "tab:green",
}


def generate_distributions(
    n: int = N,
    seed: int = SEED,
) -> dict[str, np.ndarray]:
    """Generate deterministic exponential, lognormal, and Pareto samples."""
    rng = np.random.default_rng(seed)
    exponential = rng.exponential(scale=EXPONENTIAL_SCALE, size=n)
    lognormal = rng.lognormal(mean=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA, size=n)
    u = rng.random(n)
    pareto = PARETO_XMIN * (1.0 - u) ** (-1.0 / (PARETO_ALPHA_PDF - 1.0))
    return {
        "exponential": exponential,
        "lognormal": lognormal,
        "pareto": pareto,
    }


def bin_codes_and_edges(
    values: np.ndarray,
    k: int,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return bin membership codes and exact bin edges for one method."""
    series = pd.Series(values)

    if method == "cut":
        codes, edges = pd.cut(
            series,
            bins=k,
            labels=False,
            retbins=True,
            include_lowest=True,
            duplicates="raise",
        )
    elif method == "qcut":
        codes, edges = pd.qcut(
            series,
            q=k,
            labels=False,
            retbins=True,
            duplicates="raise",
        )
    elif method == "log_binning":
        xmin = float(values.min())
        xmax = float(values.max())
        if xmin <= 0:
            raise ValueError("Logarithmic binning requires strictly positive data.")
        edges = np.geomspace(xmin, xmax, k + 1)
        edges[0] = np.nextafter(xmin, -np.inf)
        edges[-1] = np.nextafter(xmax, np.inf)
        codes = pd.cut(
            series,
            bins=edges,
            labels=False,
            include_lowest=True,
            right=True,
        )
    else:
        raise ValueError(f"Unknown binning method: {method}")

    code_array = np.asarray(codes, dtype=np.int64)
    edge_array = np.asarray(edges, dtype=np.float64)
    if len(edge_array) != k + 1:
        raise AssertionError(f"{method}: expected {k + 1} edges, got {len(edge_array)}")
    if (code_array < 0).any() or (code_array >= k).any():
        raise AssertionError(f"{method}: invalid bin assignments")
    return code_array, edge_array


def summarize_bins(
    values: np.ndarray,
    distribution: str,
    k: int,
    method: str,
) -> pd.DataFrame:
    """Compute one row of local statistics for each bin."""
    codes, edges = bin_codes_and_edges(values, k, method)
    frame = pd.DataFrame({"bin_id": codes, "value": values})
    grouped = frame.groupby("bin_id", sort=True, observed=True)["value"]

    stats = grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        max="max",
    ).reindex(range(k))
    stats["p90"] = grouped.quantile(0.90).reindex(range(k))
    stats["p99"] = grouped.quantile(0.99).reindex(range(k))

    result = stats.reset_index()
    result["bin_id"] += 1
    result["bin_left"] = edges[:-1]
    result["bin_right"] = edges[1:]
    result["bin_width"] = result["bin_right"] - result["bin_left"]
    result["bin_center"] = 0.5 * (result["bin_left"] + result["bin_right"])
    result["frequency"] = result["count"].fillna(0.0) / N
    result["density"] = result["count"].fillna(0.0) / (N * result["bin_width"])

    result.insert(0, "K", k)
    result.insert(0, "N", N)
    result.insert(0, "method", method)
    result.insert(0, "distribution", distribution)

    columns = [
        "distribution",
        "method",
        "N",
        "K",
        "bin_id",
        "bin_left",
        "bin_right",
        "bin_width",
        "bin_center",
        "count",
        "frequency",
        "density",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p90",
        "p99",
    ]
    result = result[columns]

    if int(result["count"].fillna(0).sum()) != N:
        raise AssertionError(f"Lost observations for {distribution}, K={k}, {method}.")
    return result


def empirical_ccdf(
    values: np.ndarray,
    max_points: int = 4000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a tail-resolved down-sampled empirical CCDF."""
    x = np.sort(values)
    n = len(x)
    remaining = np.unique(np.rint(np.geomspace(1, n, max_points)).astype(np.int64))
    idx = np.sort(n - remaining)
    return x[idx], (n - idx) / n


def loglog_tail_ols(
    x: np.ndarray,
    ccdf: np.ndarray,
) -> tuple[float, float, float]:
    """Illustrative OLS in log-log coordinates over 1%--50% survival."""
    mask = (x > 0) & (ccdf >= 0.01) & (ccdf <= 0.50)
    lx = np.log10(x[mask])
    ly = np.log10(ccdf[mask])
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = slope * lx + intercept
    ss_res = float(np.sum((ly - fitted) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), float(r2)


def make_histogram_grid(samples: dict[str, np.ndarray]) -> Path:
    """Create the 3x3 histogram grid."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 10.5), constrained_layout=True)

    for i, k in enumerate(K_VALUES):
        for j, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[i, j]
            values = samples[distribution]

            if distribution == "pareto":
                display_xmin = float(values.min())
                display_xmax = float(np.quantile(values, PARETO_DISPLAY_QUANTILE))
                display_values = np.minimum(values, display_xmax)
                counts, edges = np.histogram(
                    display_values,
                    bins=k,
                    range=(display_xmin, display_xmax),
                )
            else:
                counts, edges = np.histogram(values, bins=k)

            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                color=HISTOGRAM_COLORS[distribution],
                edgecolor="black",
                linewidth=0.8,
            )
            ax.margins(x=0)
            if distribution == "pareto":
                ax.set_yscale("log")
                ax.set_xlim(left=display_xmin, right=display_xmax)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.35, linestyle="--")
            ax.set_title(f"{distribution.title()} (K = {k})", fontsize=13)
            ax.set_xlabel("Income")
            ax.set_ylabel("Frequency")

    fig.savefig(FIGURE_HISTOGRAMS, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return FIGURE_HISTOGRAMS


def _draw_ccdf_grid(
    samples: dict[str, np.ndarray],
    *,
    output_path: Path,
    include_title: bool,
) -> Path:
    """Draw the common 3x3 CCDF grid."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)

    for i, k in enumerate(K_VALUES):
        for j, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[i, j]
            x, ccdf = empirical_ccdf(samples[distribution])
            slope, intercept, r2 = loglog_tail_ols(x, ccdf)

            ax.plot(x, ccdf, linewidth=1.0, label="Empirical CCDF")
            fit_mask = (x > 0) & (ccdf >= 0.01) & (ccdf <= 0.50)
            x_fit = x[fit_mask]
            y_fit = 10 ** (intercept + slope * np.log10(x_fit))
            ax.plot(x_fit, y_fit, linestyle="--", linewidth=1.0, label="OLS")

            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            ax.text(
                0.04,
                0.05,
                f"slope={slope:.3f}\n$R^2$={r2:.3f}",
                transform=ax.transAxes,
                fontsize=8,
            )

            if i == 0:
                ax.set_title(distribution.title())
            if j == 0:
                ax.set_ylabel(f"K={k}\nP(X ≥ x)")
            if i == 2:
                ax.set_xlabel("x")

    axes[0, 0].legend(loc="best", fontsize=8)
    if include_title:
        fig.suptitle(
            f"Empirical CCDF and illustrative log-log OLS: N={N:,}\n"
            "CCDF is independent of K; rows repeat it for comparison"
        )

    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_ccdf_figures(samples: dict[str, np.ndarray]) -> None:
    """Create CCDF figures with and without the embedded super-title."""
    _draw_ccdf_grid(samples, output_path=FIGURE_CCDF, include_title=True)
    _draw_ccdf_grid(samples, output_path=FIGURE_CCDF_NO_TITLE, include_title=False)


def make_bin_mean_grid(table: pd.DataFrame) -> Path:
    """Create a K-by-distribution grid of mean value versus bin id."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)

    for i, k in enumerate(K_VALUES):
        for j, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[i, j]
            for method in METHODS:
                subset = table[
                    (table["distribution"] == distribution)
                    & (table["method"] == method)
                    & (table["K"] == k)
                ].sort_values("bin_id")
                ax.plot(
                    subset["bin_id"],
                    subset["mean"],
                    linewidth=1.4,
                    label=method.replace("_", " "),
                )

            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.set_title(distribution.title())
            if j == 0:
                ax.set_ylabel(f"K={k}\nMean in bin")
            if i == 2:
                ax.set_xlabel("Bin id")
            ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"Mean value by bin and method: N={N:,}")
    fig.savefig(FIGURE_BIN_MEANS, dpi=180)
    plt.close(fig)
    return FIGURE_BIN_MEANS


def generate_pareto_sample(
    n: int = N,
    alpha: float = NEWMAN_ALPHA,
    xmin: float = NEWMAN_XMIN,
    seed: int = NEWMAN_SEED,
) -> np.ndarray:
    """Generate the continuous Pareto sample used in the Newman-style figure."""
    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1.")
    if xmin <= 0.0:
        raise ValueError("xmin must be strictly positive.")

    rng = np.random.default_rng(seed)
    u = rng.random(n)
    return xmin * (1.0 - u) ** (-1.0 / (alpha - 1.0))


def _prepare_newman_panel_data(
    values: np.ndarray,
    alpha: float,
    xmin: float,
) -> dict[str, np.ndarray]:
    """Prepare arrays used by the four Newman-style panels."""
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


def _style_newman_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="major", length=8, width=1.0)
    ax.tick_params(which="minor", length=4, width=0.8)


def _draw_newman_panel_a(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    ax.plot(data["x_pdf"], data["pdf"], linewidth=1.2)
    ax.set_xlim(0.0, float(data["x_linear_max"][0]))
    ax.set_ylim(0.0, 1.5 * float(data["normalization"][0]))
    ax.set_xlabel(r"$x$")
    ax.set_ylabel("samples")
    ax.text(0.56, 0.88, "(a)", transform=ax.transAxes, fontsize=15)
    _style_newman_axis(ax)


def _draw_newman_panel_b(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
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
    _style_newman_axis(ax)


def _draw_newman_panel_c(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
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
    _style_newman_axis(ax)


def _draw_newman_panel_d(ax: plt.Axes, data: dict[str, np.ndarray]) -> None:
    ax.plot(data["ccdf_x"], data["ccdf"], linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"samples with value $>x$")
    ax.text(0.56, 0.88, "(d)", transform=ax.transAxes, fontsize=15)
    _style_newman_axis(ax)


NEWMAN_PANEL_DRAWERS = (
    _draw_newman_panel_a,
    _draw_newman_panel_b,
    _draw_newman_panel_c,
    _draw_newman_panel_d,
)


def _save_newman_panels(data: dict[str, np.ndarray]) -> None:
    """Save the four Newman-style component panels in the project root folder."""
    for label, drawer in zip("abcd", NEWMAN_PANEL_DRAWERS):
        fig, ax = plt.subplots(figsize=(5.1, 4.0))
        drawer(ax, data)
        fig.tight_layout()
        fig.savefig(
            ROOT / f"figure_newman_panel_{label}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)


def make_newman_powerlaw_grid(
    values: np.ndarray | None = None,
    *,
    alpha: float = NEWMAN_ALPHA,
    xmin: float = NEWMAN_XMIN,
    show: bool = False,
) -> Path:
    """Create the Newman-style 2x2 power-law grid and its four component panels."""
    if values is None:
        values = generate_pareto_sample(alpha=alpha, xmin=xmin)
    else:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("values must be a non-empty one-dimensional array.")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("values must contain only finite positive numbers.")

    data = _prepare_newman_panel_data(values, alpha, xmin)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6))
    for ax, drawer in zip(axes.ravel(), NEWMAN_PANEL_DRAWERS):
        drawer(ax, data)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.10,
        top=0.98,
        wspace=0.36,
        hspace=0.34,
    )
    fig.savefig(FIGURE_NEWMAN_GRID, dpi=200, bbox_inches="tight")
    _save_newman_panels(data)

    if show:
        plt.show()
    plt.close(fig)
    return FIGURE_NEWMAN_GRID


def main() -> None:
    """Generate the table and every figure in one reproducible run."""
    samples = generate_distributions()
    tables = [
        summarize_bins(samples[distribution], distribution, k, method)
        for distribution in DISTRIBUTIONS
        for k in K_VALUES
        for method in METHODS
    ]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(TABLE_PATH, index=False, float_format="%.12g")

    expected_rows = len(DISTRIBUTIONS) * len(METHODS) * sum(K_VALUES)
    if len(table) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(table)}")

    histogram_samples = generate_distributions(n=HISTOGRAM_N)
    make_histogram_grid(histogram_samples)
    make_ccdf_figures(samples)
    make_bin_mean_grid(table)
    make_newman_powerlaw_grid()

    print(f"Saved {len(table):,} bin rows to {TABLE_PATH.name}")
    print("Generated all figure_*.png outputs in log_binning")


if __name__ == "__main__":
    main()
