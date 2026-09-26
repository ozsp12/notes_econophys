"""Run compact binning experiments for exponential, lognormal, and Pareto data.

A single synthetic sample size N=1,000,000 is used throughout. Raw samples are
generated in memory and are not persisted. The long CSV stores one row per bin
for cut, qcut, and logarithmic binning. Histograms use conventional equal-width
bins over the complete sample, absolute frequency, and a logarithmic y-axis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "binning_experiments.csv"

SEED = 20260925
N = 1_000_000
K_VALUES = (25, 50, 100)
DISTRIBUTIONS = ("exponential", "lognormal", "pareto")
METHODS = ("cut", "qcut", "log_binning")

# Synthetic scales are chosen only to put the three examples on readable
# income-like horizontal ranges. They do not change the distribution families.
EXPONENTIAL_SCALE = 50.0
LOGNORMAL_MU = float(np.log(150.0))
LOGNORMAL_SIGMA = 0.55
PARETO_ALPHA_PDF = 2.5
PARETO_XMIN = 100.0

HISTOGRAM_COLORS = {
    "exponential": "tab:blue",
    "lognormal": "tab:orange",
    "pareto": "tab:green",
}


def generate_distributions() -> dict[str, np.ndarray]:
    """Generate the three deterministic synthetic samples."""
    rng = np.random.default_rng(SEED)
    exponential = rng.exponential(scale=EXPONENTIAL_SCALE, size=N)
    lognormal = rng.lognormal(mean=LOGNORMAL_MU, sigma=LOGNORMAL_SIGMA, size=N)

    # p(x) = (alpha - 1) x_min^(alpha - 1) x^(-alpha), x >= x_min.
    u = rng.random(N)
    pareto = PARETO_XMIN * (1.0 - u) ** (-1.0 / (PARETO_ALPHA_PDF - 1.0))

    return {
        "exponential": exponential,
        "lognormal": lognormal,
        "pareto": pareto,
    }


def bin_codes_and_edges(values: np.ndarray, k: int, method: str) -> tuple[np.ndarray, np.ndarray]:
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
    """Compute one row of local statistics for each of the K bins."""
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


def empirical_ccdf(values: np.ndarray, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Return a tail-resolved down-sampled empirical CCDF."""
    x = np.sort(values)
    n = len(x)
    remaining = np.unique(np.rint(np.geomspace(1, n, max_points)).astype(np.int64))
    idx = np.sort(n - remaining)
    return x[idx], (n - idx) / n


def loglog_tail_ols(x: np.ndarray, ccdf: np.ndarray) -> tuple[float, float, float]:
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


def make_histogram_grid(samples: dict[str, np.ndarray]) -> None:
    """Create the 3x3 full-sample histogram grid for K=25, 50, and 100."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), constrained_layout=True)

    for i, k in enumerate(K_VALUES):
        for j, distribution in enumerate(DISTRIBUTIONS):
            ax = axes[i, j]
            values = samples[distribution]

            # Use the complete sample. No clipping, truncation, or fixed range.
            counts, edges = np.histogram(values, bins=k)

            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                color=HISTOGRAM_COLORS[distribution],
                edgecolor="black",
                linewidth=0.55,
            )
            ax.set_yscale("log")
            ax.set_ylim(bottom=1)
            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.set_title(f"{distribution.title()} (K = {k})", fontsize=13)
            ax.set_xlabel("Income")
            ax.set_ylabel("Frequency (log)")

    fig.suptitle(
        "Synthetic Income Histograms\n"
        f"N = {N:,} observations; equal-width bins; y-axis in log scale",
        fontsize=20,
    )
    fig.savefig(ROOT / "histograms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_ccdf_grid(samples: dict[str, np.ndarray]) -> None:
    """Create a K-by-distribution CCDF grid; rows repeat the K-independent CCDF."""
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
    fig.suptitle(
        f"Empirical CCDF and illustrative log-log OLS: N={N:,}\n"
        "CCDF is independent of K; rows repeat it for comparison"
    )
    fig.savefig(ROOT / "ccdf_grid.png", dpi=180)
    plt.close(fig)


def make_bin_mean_grid(table: pd.DataFrame) -> None:
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

            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            if i == 0:
                ax.set_title(distribution.title())
            if j == 0:
                ax.set_ylabel(f"K={k}\nMean in bin")
            if i == 2:
                ax.set_xlabel("Bin id")

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"Mean value by bin and method: N={N:,}")
    fig.savefig(ROOT / "bin_means_grid.png", dpi=180)
    plt.close(fig)


def main() -> None:
    samples = generate_distributions()

    tables = [
        summarize_bins(samples[distribution], distribution, k, method)
        for distribution in DISTRIBUTIONS
        for k in K_VALUES
        for method in METHODS
    ]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(CSV_PATH, index=False, float_format="%.12g")

    expected_rows = len(DISTRIBUTIONS) * len(METHODS) * sum(K_VALUES)
    if len(table) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} rows, got {len(table)}")

    make_histogram_grid(samples)
    make_ccdf_grid(samples)
    make_bin_mean_grid(table)

    print(f"Saved {len(table):,} bin rows to {CSV_PATH.name}")
    print("Generated 3 figures")


if __name__ == "__main__":
    main()
