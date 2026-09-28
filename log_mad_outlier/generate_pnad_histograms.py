"""Generate the PNAD histogram figures used in the log-MAD teaching note.

The script compares the same four survey years in the refined and trusted
layers of ``ozsp12/project_pnad``.  The refined sample defines the histogram
bin edges for each year; the trusted sample is then plotted with exactly the
same edges, making the before/after comparison direct.

By default the script downloads only the required PNAD files from GitHub.  A
local clone of ``project_pnad`` can instead be supplied with
``--project-pnad-root``.

Outputs, written next to this script unless ``--output-dir`` is supplied:

    figure_pnad_refined_histograms.png
    figure_pnad_trusted_histograms.png

Dependencies: numpy, pandas, matplotlib, pyarrow.
"""

from __future__ import annotations

import argparse
import tempfile
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_PNAD_RAW = "https://raw.githubusercontent.com/ozsp12/project_pnad/main"
YEARS = (1979, 1985, 1989, 1990)
N_BINS = 100

REFINED_FIGURE = "figure_pnad_refined_histograms.png"
TRUSTED_FIGURE = "figure_pnad_trusted_histograms.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate refined and trusted PNAD histograms for selected years."
    )
    parser.add_argument(
        "--project-pnad-root",
        type=Path,
        default=None,
        help="Optional path to a local clone of ozsp12/project_pnad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory in which the PNG figures are written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help=f"Number of equal-width bins per year (default: {N_BINS}).",
    )
    return parser.parse_args()


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def _metadata_path(project_root: Path | None, cache_dir: Path) -> Path:
    relative = Path("data/analytics/pnad_annual_metadata.csv")
    if project_root is not None:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        return path
    return _download(
        f"{PROJECT_PNAD_RAW}/{relative.as_posix()}",
        cache_dir / relative.name,
    )


def _parquet_path(
    layer: str,
    year: int,
    project_root: Path | None,
    cache_dir: Path,
) -> Path:
    filename = f"pnad_{layer}_{year}.parquet"
    relative = Path("data") / layer / filename
    if project_root is not None:
        path = project_root / relative
        if not path.exists():
            raise FileNotFoundError(f"PNAD file not found: {path}")
        return path
    return _download(
        f"{PROJECT_PNAD_RAW}/{relative.as_posix()}",
        cache_dir / filename,
    )


def load_metadata(project_root: Path | None, cache_dir: Path) -> pd.DataFrame:
    metadata = pd.read_csv(_metadata_path(project_root, cache_dir))
    required = {"ano", "exchange", "inflation_factor_2025"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            "Missing metadata column(s): " + ", ".join(sorted(missing))
        )
    metadata = metadata.copy()
    metadata["ano"] = pd.to_numeric(metadata["ano"], errors="raise").astype(int)
    return metadata.set_index("ano", drop=False)


def load_adjusted_income(
    layer: str,
    year: int,
    metadata: pd.DataFrame,
    project_root: Path | None,
    cache_dir: Path,
) -> np.ndarray:
    """Load one annual layer and express income on the common 2025 USD scale."""
    parquet = _parquet_path(layer, year, project_root, cache_dir)
    df = pd.read_parquet(parquet, columns=["renda"])

    income = pd.to_numeric(df["renda"], errors="coerce").to_numpy(dtype=float)
    income = income[np.isfinite(income) & (income >= 0)]
    if income.size == 0:
        raise ValueError(f"{layer} {year}: no finite non-negative income values")

    row = metadata.loc[year]
    exchange = float(row["exchange"])
    inflation = float(row["inflation_factor_2025"])
    if not np.isfinite(exchange) or exchange <= 0:
        raise ValueError(f"{year}: invalid exchange factor {exchange}")
    if not np.isfinite(inflation) or inflation <= 0:
        raise ValueError(f"{year}: invalid inflation factor {inflation}")

    return income / exchange * inflation


def histogram_edges(values: np.ndarray, bins: int) -> np.ndarray:
    """Return equal-width edges, matching the PNAD publication histogram logic."""
    if bins < 1:
        raise ValueError("bins must be a positive integer")
    return np.histogram_bin_edges(values, bins=bins)


def plot_layer(
    data: dict[int, np.ndarray],
    edges_by_year: dict[int, np.ndarray],
    layer: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), constrained_layout=True)

    panel_labels = ("(a)", "(b)", "(c)", "(d)")
    for ax, label, year in zip(axes.ravel(), panel_labels, YEARS):
        values = data[year]
        edges = edges_by_year[year]
        counts, _ = np.histogram(values, bins=edges)

        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            linewidth=0.35,
        )
        ax.set_yscale("log")
        ax.set_xlim(edges[0], edges[-1])
        ax.set_title(f"{label} {year}")
        ax.set_xlabel("Income (2025 USD)")
        ax.set_ylabel("Frequency")
        ax.grid(axis="y", linestyle="--", alpha=0.25)

    fig.suptitle(f"PNAD {layer} income distributions", fontsize=15)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_root = (
        args.project_pnad_root.resolve() if args.project_pnad_root is not None else None
    )

    with tempfile.TemporaryDirectory(prefix="pnad_histograms_") as tmp:
        cache_dir = Path(tmp)
        metadata = load_metadata(project_root, cache_dir)

        refined = {
            year: load_adjusted_income(
                "refined", year, metadata, project_root, cache_dir
            )
            for year in YEARS
        }
        trusted = {
            year: load_adjusted_income(
                "trusted", year, metadata, project_root, cache_dir
            )
            for year in YEARS
        }

        # The refined sample fixes the bin edges.  Reusing those exact edges for
        # trusted makes the two figures directly comparable year by year.
        edges_by_year = {
            year: histogram_edges(refined[year], args.bins) for year in YEARS
        }

        refined_path = output_dir / REFINED_FIGURE
        trusted_path = output_dir / TRUSTED_FIGURE

        plot_layer(refined, edges_by_year, "refined", refined_path)
        plot_layer(trusted, edges_by_year, "trusted", trusted_path)

    print(refined_path)
    print(trusted_path)


if __name__ == "__main__":
    main()
