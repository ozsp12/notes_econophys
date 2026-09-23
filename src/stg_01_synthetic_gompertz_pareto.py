"""Build synthetic PNAD income microdata from annual Gompertz--Pareto fits.

The annual metadata contain parameters fitted to positive income in the trusted
PNAD analysis. The reconstruction uses deterministic midpoint survival quantiles,
so no Monte Carlo noise is introduced.

For normalized income x,

    S_G(x) = exp(exp(A - B x))

for the Gompertz body, and

    S_P(x) = beta x**(-alpha)

for the Pareto tail, with S expressed in percent. Continuity at the transition
x_t gives beta = S_G(x_t) * x_t**alpha.

Generated normalized incomes are mapped to 2025 USD with the empirical annual
normalization mean. The resulting data are synthetic fitted-distribution data,
not recovered original PNAD microdata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("year", pa.int16(), nullable=False),
        pa.field("income", pa.float64(), nullable=False),
    ]
)


def load_metadata(path: Path = DEFAULT_METADATA) -> pd.DataFrame:
    """Load and validate annual Gompertz--Pareto reconstruction metadata."""
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if df["year"].duplicated().any():
        duplicated = df.loc[df["year"].duplicated(), "year"].tolist()
        raise ValueError(f"Duplicated years in metadata: {duplicated}")

    for column in (
        "normalization_mean_income_adj_2025_usd",
        "positive_income_observation_n",
        "gompertz_B",
    ):
        if (df[column] <= 0).any():
            raise ValueError(f"{column} must be strictly positive.")

    return df


def interpolate_missing_tail_parameters(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate only missing tail parameters between observed survey years.

    In the source fit, 1985 has no Pareto transition or tail exponent. The
    interpolation is explicit and local; observed fitted values are not changed.
    This function does not create rows for years in which PNAD was not fielded.
    """
    result = df.copy()

    for column in ("transition_x_t", "pareto_alpha_mle"):
        source_missing = result[column].isna()
        result[column] = result[column].interpolate(
            method="linear",
            limit_area="inside",
        )
        result[f"{column}_interpolated"] = (
            source_missing & result[column].notna()
        )

    unresolved = result[["transition_x_t", "pareto_alpha_mle"]].isna().any(axis=1)
    if unresolved.any():
        years = result.loc[unresolved, "year"].tolist()
        raise ValueError(
            "Tail parameters remain unavailable after interpolation for "
            f"years {years}."
        )

    return result


def reconstruct_year(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic year and income arrays for one annual fitted model."""
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

    rank = np.arange(1, n + 1, dtype=np.float64)
    survival_pct = 100.0 * (n - rank + 0.5) / n

    transition_survival_pct = float(np.exp(np.exp(A - B * x_t)))
    if not 1.0 <= transition_survival_pct <= 100.0:
        raise ValueError(
            f"{year}: transition survival must lie in [1, 100] percent; "
            f"got {transition_survival_pct}."
        )

    normalized_income = np.empty(n, dtype=np.float64)
    body = survival_pct >= transition_survival_pct
    tail = ~body

    normalized_income[body] = (
        A - np.log(np.log(survival_pct[body]))
    ) / B

    beta = transition_survival_pct * x_t**alpha
    normalized_income[tail] = (
        beta / survival_pct[tail]
    ) ** (1.0 / alpha)

    normalized_income = np.maximum(normalized_income, 0.0)
    income = normalized_income * mean_income

    if not np.isfinite(income).all() or (income < 0).any():
        raise ValueError(f"{year}: reconstruction produced invalid incomes.")

    years = np.full(n, year, dtype=np.int16)
    return years, income.astype(np.float64, copy=False)


def annual_arrow_table(row: pd.Series) -> pa.Table:
    """Convert one reconstructed survey year to the canonical Arrow schema."""
    years, income = reconstruct_year(row)
    return pa.Table.from_arrays(
        [
            pa.array(years, type=pa.int16()),
            pa.array(income, type=pa.float64()),
        ],
        schema=PARQUET_SCHEMA,
    )


def build_synthetic_dataset(
    metadata_path: Path = DEFAULT_METADATA,
    output_path: Path = DEFAULT_OUTPUT,
    interpolate_missing_tail: bool = True,
) -> tuple[int, int]:
    """Write the complete synthetic dataset to Parquet one survey year at a time."""
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    rows_written = 0
    years_written = 0

    with pq.ParquetWriter(
        output_path,
        PARQUET_SCHEMA,
        compression="zstd",
    ) as writer:
        for _, row in metadata.iterrows():
            table = annual_arrow_table(row)
            writer.write_table(table)
            rows_written += table.num_rows
            years_written += 1

    expected_rows = int(metadata["positive_income_observation_n"].sum())
    if rows_written != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows, wrote {rows_written}."
        )

    return rows_written, years_written


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
    rows, years = build_synthetic_dataset(
        metadata_path=args.metadata,
        output_path=args.output,
        interpolate_missing_tail=not args.no_tail_interpolation,
    )
    print(f"Saved {rows:,} rows for {years} years to {args.output}")


if __name__ == "__main__":
    main()
