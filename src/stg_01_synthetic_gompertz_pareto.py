"""Build synthetic PNAD income microdata from annual Gompertz--Pareto fits.

The source metadata stores the parameters fitted to positive annual income in the
trusted PNAD analysis. Income is reconstructed deterministically from midpoint
survival quantiles. This produces a smooth synthetic sample with the same number
of observations as the fitted annual sample and avoids Monte Carlo noise.

For normalized income x, the fitted survival function is

    S_G(x) = exp(exp(A - B x))               (Gompertz body)

and, above the transition x_t,

    S_P(x) = beta x**(-alpha)                (Pareto tail),

where S is expressed in percent. Continuity at x_t implies

    beta = S_G(x_t) * x_t**alpha.

The generated normalized incomes are mapped back to 2025 USD using the empirical
annual normalization mean stored in the metadata.

This is a synthetic reconstruction of the fitted distributions. It is not an
inverse recovery of the original PNAD records.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = REPO_ROOT / "data" / "metadata" / "pnad_gompertz_pareto.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "synthetic" / "pnad_gompertz_pareto.parquet"

REQUIRED_COLUMNS = {
    "year",
    "normalization_mean_income_adj_2025_usd",
    "positive_income_observation_n",
    "gompertz_A",
    "gompertz_B",
    "transition_x_t",
    "pareto_alpha_mle",
}


def load_metadata(path: Path = DEFAULT_METADATA) -> pd.DataFrame:
    """Load and validate annual Gompertz--Pareto reconstruction metadata."""
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if df["year"].duplicated().any():
        duplicated = df.loc[df["year"].duplicated(), "year"].tolist()
        raise ValueError(f"Duplicated years in metadata: {duplicated}")

    positive_columns = [
        "normalization_mean_income_adj_2025_usd",
        "positive_income_observation_n",
        "gompertz_B",
    ]
    for column in positive_columns:
        if (df[column] <= 0).any():
            raise ValueError(f"{column} must be strictly positive.")

    return df


def interpolate_missing_tail_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate only missing tail parameters between observed PNAD years.

    The source fit has no supported Pareto tail for 1985. The interpolation is
    explicit and local; observed values are never modified. This function does
    not create rows for years without a PNAD survey.
    """
    result = df.copy()
    columns = ["transition_x_t", "pareto_alpha_mle"]

    for column in columns:
        original_missing = result[column].isna()
        result[column] = result[column].interpolate(
            method="linear",
            limit_area="inside",
        )
        result[f"{column}_interpolated"] = original_missing & result[column].notna()

    unresolved = result[columns].isna().any(axis=1)
    if unresolved.any():
        years = result.loc[unresolved, "year"].tolist()
        raise ValueError(
            "Tail parameters remain unavailable after interpolation for years "
            f"{years}."
        )

    return result


def reconstruct_year(row: pd.Series) -> pd.DataFrame:
    """Reconstruct one annual synthetic sample by inverse survival quantiles."""
    year = int(row["year"])
    n = int(row["positive_income_observation_n"])
    mean_income = float(row["normalization_mean_income_adj_2025_usd"])
    A = float(row["gompertz_A"])
    B = float(row["gompertz_B"])
    x_t = float(row["transition_x_t"])
    alpha = float(row["pareto_alpha_mle"])

    if n <= 0:
        raise ValueError(f"{year}: observation count must be positive.")
    if B <= 0 or x_t <= 0 or alpha <= 0:
        raise ValueError(f"{year}: invalid Gompertz--Pareto parameters.")

    # Midpoint empirical survival probabilities, ordered from low to high income.
    rank = np.arange(1, n + 1, dtype=np.float64)
    survival_pct = 100.0 * (n - rank + 0.5) / n

    transition_survival_pct = np.exp(np.exp(A - B * x_t))
    if not (1.0 <= transition_survival_pct <= 100.0):
        raise ValueError(
            f"{year}: transition survival must lie in [1, 100] percent; "
            f"got {transition_survival_pct}."
        )

    normalized_income = np.empty(n, dtype=np.float64)
    body = survival_pct >= transition_survival_pct
    tail = ~body

    # Invert S_G(x) = exp(exp(A - Bx)).
    normalized_income[body] = (
        A - np.log(np.log(survival_pct[body]))
    ) / B

    # Enforce continuity and invert S_P(x) = beta x^(-alpha).
    beta = transition_survival_pct * x_t**alpha
    normalized_income[tail] = (beta / survival_pct[tail]) ** (1.0 / alpha)

    # Numerical roundoff near S=100 can produce values extremely close to zero.
    normalized_income = np.maximum(normalized_income, 0.0)
    income = normalized_income * mean_income

    return pd.DataFrame(
        {
            "year": np.full(n, year, dtype=np.int16),
            "income": income.astype(np.float64),
        }
    )


def build_synthetic_dataset(
    metadata_path: Path = DEFAULT_METADATA,
    output_path: Path = DEFAULT_OUTPUT,
    interpolate_missing_tail: bool = True,
) -> pd.DataFrame:
    """Build all available PNAD survey years and persist a two-column Parquet."""
    metadata = load_metadata(metadata_path)

    if interpolate_missing_tail:
        metadata = interpolate_missing_tail_parameters(metadata)
    elif metadata[["transition_x_t", "pareto_alpha_mle"]].isna().any(axis=None):
        years = metadata.loc[
            metadata[["transition_x_t", "pareto_alpha_mle"]].isna().any(axis=1),
            "year",
        ].tolist()
        raise ValueError(
            "Missing tail parameters. Enable interpolation or remove affected "
            f"years: {years}"
        )

    frames = [reconstruct_year(row) for _, row in metadata.iterrows()]
    synthetic = pd.concat(frames, ignore_index=True)

    if list(synthetic.columns) != ["year", "income"]:
        raise AssertionError("Synthetic dataset schema must be exactly year, income.")
    if synthetic["income"].isna().any() or (synthetic["income"] < 0).any():
        raise AssertionError("Synthetic income contains invalid values.")

    expected_rows = int(metadata["positive_income_observation_n"].sum())
    if len(synthetic) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows, generated {len(synthetic)}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synthetic.to_parquet(output_path, index=False, compression="zstd")
    return synthetic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct synthetic PNAD Gompertz--Pareto income data."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="Annual reconstruction metadata CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Parquet path.",
    )
    parser.add_argument(
        "--no-tail-interpolation",
        action="store_true",
        help="Fail instead of interpolating missing tail parameters such as 1985.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic = build_synthetic_dataset(
        metadata_path=args.metadata,
        output_path=args.output,
        interpolate_missing_tail=not args.no_tail_interpolation,
    )
    print(
        f"Saved {len(synthetic):,} rows for "
        f"{synthetic['year'].nunique()} years to {args.output}"
    )


if __name__ == "__main__":
    main()
