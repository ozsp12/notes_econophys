"""Generate manuscript-specific figure assets for the log-binning lecture notes.

The numerical definitions are imported from ``stg_01_run_experiments`` so that
manuscript figures use the same deterministic samples, CCDF construction, and
tail-fit convention as the main experiment pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stg_01_run_experiments import (
    DISTRIBUTIONS,
    K_VALUES,
    empirical_ccdf,
    generate_distributions,
    loglog_tail_ols,
)


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
CCDF_ASSET_PATH = ASSET_DIR / "ccdf_grid_no_title.png"


def make_ccdf_grid_no_title(
    samples: dict[str, np.ndarray] | None = None,
    *,
    output_path: str | Path = CCDF_ASSET_PATH,
) -> Path:
    """Create the manuscript CCDF grid without an embedded super-title."""
    if samples is None:
        samples = generate_distributions()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    path = make_ccdf_grid_no_title()
    print(f"Generated {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
