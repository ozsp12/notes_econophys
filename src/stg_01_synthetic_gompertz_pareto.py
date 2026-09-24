"""Build synthetic PNAD income microdata from Gompertz--Pareto fits and moments.

The annual Gompertz--Pareto metadata define the shape of each income
distribution. Years in which PNAD was not fielded are represented by the simple
arithmetic mean of the immediately preceding and following years.

The trusted ``statistics_annual.csv`` table supplies empirical annual moments.
Missing survey years are reconstructed in memory with the same adjacent-year
arithmetic-mean rule. The synthetic data use:

* ``income_observation_n`` for the annual sample size;
* ``income_mean_2025_usd`` for the annual mean;
* ``income_std_2025_usd`` for the annual dispersion;
* ``income_median_2025_usd``, ``Gini`` and the top-share statistics as
  validation targets retained in metadata.

For normalized income x, the fitted survival model is

    S_G(x) = exp(exp(A - B x))

for the Gompertz body and

    S_P(x) = beta x**(-alpha)

for the Pareto tail, with S expressed in percent and continuity imposed at x_t.
Deterministic midpoint survival quantiles define the raw synthetic ranking. A
monotone power calibration then matches the trusted annual mean and standard
deviation while preserving that ranking and the Gompertz--Pareto shape.

The resulting observations are synthetic fitted-distribution data, not recovered
original PNAD microdata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISTRIBUTION_METADATA = (
    REPO_ROOT / "data" / "metadata" / "pnad_gompertz_pareto.csv"
)
DEFAULT_STATISTICS = REPO_ROOT / "data" / "metadata" / "statistics_annual.csv"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "synthetic" / "pnad_gompertz_pareto.parquet"

FIRST_YEAR = 1976
LAST_YEAR = 2025

DISTRIBUTION_COLUMNS = {
    "year",
    "gompertz_A",
    "gompertz_B",
    "transition_x_t",
    "pareto_alpha_mle",
}

STATISTICS_COLUMNS = {
    "year",
    "income_observation_n",
    "income_mean_2025_usd",
    "income_median_2025_usd",
    "income_std_2025_usd",
}

PARQUET_SCHEMA = pa.schema(
    [
        pa.field("year", pa.int16(), nullable=False),
        pa.field("income", pa.float64(), nullable=False),
    ]
)


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def adjacent_year_mean_rows(
    df: pd.DataFrame,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
) -> pd.DataFrame:
    """Insert missing years as arithmetic means of y-1 and y+1.

    The rule is intentionally local and simple. It is used only for isolated
    years without PNAD: 1980, 1991, 1994, 2000 and 2010. Numeric columns are
    averaged. Non-numeric columns are left missing for the synthetic model.
    """
    if "year" not in df.columns:
        raise ValueError("Metadata must contain a year column.")

    result = df.copy().sort_values("year").reset_index(drop=True)
    result["year"] = result["year"].astype(int)

    if result["year"].duplicated().any():
        duplicates = result.loc[result["year"].duplicated(), "year"].tolist()
        raise ValueError(f"Duplicated years: {duplicates}")

    result = result.set_index("year")
    numeric_columns = result.select_dtypes(include=[np.number]).columns.tolist()

    missing_years = [
        year for year in range(first_year, last_year + 1) if year not in result.index
    ]

    for year in missing_years:
        previous_year = year - 1
        next_year = year + 1
        if previous_year not in result.index or next_year not in result.index:
            raise ValueError(
                f"Cannot fill {year}: both {previous_year} and {next_year} "
                "must be present."
            )

        new_row = {column: np.nan for column in result.columns}
        for column in numeric_columns:
            previous = result.at[previous_year, column]
            following = result.at[next_year, column]
            if pd.notna(previous) and pd.notna(following):
                new_row[column] = (float(previous) + float(following)) / 2.0

        result.loc[year] = new_row

    return result.sort_index().reset_index()


def load_distribution_metadata(
    path: Path = DEFAULT_DISTRIBUTION_METADATA,
) -> pd.DataFrame:
    """Load the annual Gompertz--Pareto parameters."""
    df = pd.read_csv(path).sort_values("year").reset_index(drop=True)
    _require_columns(df, DISTRIBUTION_COLUMNS, "distribution metadata")

    expected = list(range(FIRST_YEAR, LAST_YEAR + 1))
    if df["year"].astype(int).tolist() != expected:
        raise ValueError(
            "pnad_gompertz_pareto.csv must contain every year from "
            f"{FIRST_YEAR} through {LAST_YEAR}."
        )

    for column in ("gompertz_B", "transition_x_t", "pareto_alpha_mle"):
        if df[column].isna().any() or (df[column] <= 0).any():
            raise ValueError(f"Invalid values in {column}.")

    return df


def load_statistics(path: Path = DEFAULT_STATISTICS) -> pd.DataFrame:
    """Load trusted statistics and fill non-survey years by adjacent means."""
    df = pd.read_csv(path)
    _require_columns(df, STATISTICS_COLUMNS, "statistics metadata")
    df = adjacent_year_mean_rows(df)

    for column in (
        "income_observation_n",
        "income_mean_2025_usd",
        "income_std_2025_usd",
    ):
        if df[column].isna().any() or (df[column] <= 0).any():
            raise ValueError(f"Invalid values in {column} after interpolation.")

    return df


def build_model_metadata(
    distribution_path: Path = DEFAULT_DISTRIBUTION_METADATA,
    statistics_path: Path = DEFAULT_STATISTICS,
) -> pd.DataFrame:
    """Merge distribution parameters with empirical statistical targets."""
    distribution = load_distribution_metadata(distribution_path)
    statistics = load_statistics(statistics_path)

    model = distribution.merge(
        statistics,
        on="year",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_statistics"),
    )

    expected = list(range(FIRST_YEAR, LAST_YEAR + 1))
    if model["year"].astype(int).tolist() != expected:
        raise AssertionError("Merged model metadata does not cover 1976--2025.")

    return model


def gompertz_pareto_quantiles(row: pd.Series, n: int) -> np.ndarray:
    """Return deterministic normalized-income quantiles for one year."""
    A = float(row["gompertz_A"])
    B = float(row["gompertz_B"])
    x_t = float(row["transition_x_t"])
    alpha = float(row["pareto_alpha_mle"])

    rank = np.arange(1, n + 1, dtype=np.float64)
    survival_pct = 100.0 * (n - rank + 0.5) / n

    transition_survival_pct = float(np.exp(np.exp(A - B * x_t)))
    if not 1.0 <= transition_survival_pct <= 100.0:
        raise ValueError(
            f"{int(row['year'])}: invalid transition survival "
            f"{transition_survival_pct}."
        )

    x = np.empty(n, dtype=np.float64)
    body = survival_pct >= transition_survival_pct
    tail = ~body

    x[body] = (A - np.log(np.log(survival_pct[body]))) / B

    beta = transition_survival_pct * x_t**alpha
    x[tail] = (beta / survival_pct[tail]) ** (1.0 / alpha)

    return np.maximum(x, 0.0)


def _coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(values.mean())
    if mean <= 0:
        return np.inf
    return float(values.std(ddof=0) / mean)


def calibrate_mean_and_std(
    raw_income: np.ndarray,
    target_mean: float,
    target_std: float,
) -> np.ndarray:
    """Monotonically calibrate a raw distribution to target mean and std.

    A positive power transform controls the coefficient of variation, while a
    final scale factor fixes the arithmetic mean. This preserves the rank order
    and therefore the annual Gompertz--Pareto quantile structure.
    """
    if target_mean <= 0 or target_std <= 0:
        raise ValueError("Target mean and standard deviation must be positive.")

    base = np.maximum(raw_income.astype(np.float64, copy=False), 1e-12)
    target_cv = target_std / target_mean

    def cv_at(power: float) -> float:
        return _coefficient_of_variation(np.power(base, power))

    low = 0.05
    high = 8.0
    cv_low = cv_at(low)
    cv_high = cv_at(high)

    if target_cv <= cv_low:
        power = low
    elif target_cv >= cv_high:
        power = high
    else:
        for _ in range(36):
            mid = 0.5 * (low + high)
            if cv_at(mid) < target_cv:
                low = mid
            else:
                high = mid
        power = 0.5 * (low + high)

    calibrated = np.power(base, power)
    calibrated *= target_mean / calibrated.mean()
    return calibrated


def reconstruct_year(row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct and statistically calibrate one annual synthetic sample."""
    year = int(row["year"])
    n = int(np.rint(float(row["income_observation_n"])))
    target_mean = float(row["income_mean_2025_usd"])
    target_std = float(row["income_std_2025_usd"])

    if n <= 0:
        raise ValueError(f"{year}: observation count must be positive.")

    raw = gompertz_pareto_quantiles(row, n)
    income = calibrate_mean_and_std(raw, target_mean, target_std)

    if not np.isfinite(income).all() or (income < 0).any():
        raise ValueError(f"{year}: reconstruction produced invalid incomes.")

    # Mean is fixed algebraically; the CV calibration targets the empirical std.
    if not np.isclose(income.mean(), target_mean, rtol=1e-10, atol=1e-10):
        raise AssertionError(f"{year}: synthetic mean calibration failed.")

    years = np.full(n, year, dtype=np.int16)
    return years, income.astype(np.float64, copy=False)


def annual_arrow_table(row: pd.Series) -> pa.Table:
    """Convert one reconstructed year to the canonical Arrow schema."""
    years, income = reconstruct_year(row)
    return pa.Table.from_arrays(
        [
            pa.array(years, type=pa.int16()),
            pa.array(income, type=pa.float64()),
        ],
        schema=PARQUET_SCHEMA,
    )


def build_synthetic_dataset(
    distribution_path: Path = DEFAULT_DISTRIBUTION_METADATA,
    statistics_path: Path = DEFAULT_STATISTICS,
    output_path: Path = DEFAULT_OUTPUT,
) -> tuple[int, int]:
    """Write 1976--2025 synthetic PNAD income data to one Parquet file."""
    metadata = build_model_metadata(distribution_path, statistics_path)

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

    expected_rows = int(np.rint(metadata["income_observation_n"]).sum())
    if rows_written != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows, wrote {rows_written}."
        )

    return rows_written, years_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build statistically calibrated synthetic PNAD Gompertz--Pareto data."
        )
    )
    parser.add_argument(
        "--distribution-metadata",
        type=Path,
        default=DEFAULT_DISTRIBUTION_METADATA,
        help="Annual Gompertz--Pareto metadata CSV.",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=DEFAULT_STATISTICS,
        help="Trusted annual statistics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output Parquet path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, years = build_synthetic_dataset(
        distribution_path=args.distribution_metadata,
        statistics_path=args.statistics,
        output_path=args.output,
    )
    print(f"Saved {rows:,} rows for {years} years to {args.output}")


if __name__ == "__main__":
    main()
