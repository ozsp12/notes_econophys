# notes_econophys

Lecture notes and reproducible computational material for econophysics.

## Data structure

- `data/metadata/pnad_gompertz_pareto.csv`: annual Gompertz--Pareto parameters for 1976--2025. Years without PNAD are filled by the arithmetic mean of the immediately previous and following years.
- `data/metadata/statistics_annual.csv`: trusted annual PNAD descriptive and inequality statistics copied from `project_pnad`.
- `data/metadata/monetary_metadata.csv`: Brazilian currency, exchange-rate and U.S. CPI metadata.
- `data/synthetic/`: generated synthetic datasets.

## Synthetic PNAD reconstruction

`src/stg_01_synthetic_gompertz_pareto.py` reconstructs deterministic annual income quantiles from the Gompertz--Pareto model and calibrates each synthetic year to the trusted annual observation count, mean and standard deviation.

For years without PNAD (1980, 1991, 1994, 2000 and 2010), the statistical metadata are generated in memory by the arithmetic mean of years `y-1` and `y+1`, using the same rule adopted for the Gompertz--Pareto parameters.

Run:

```bash
pip install -r requirements.txt
python src/stg_01_synthetic_gompertz_pareto.py
```

Output:

```text
data/synthetic/pnad_gompertz_pareto.parquet
```

The Parquet schema is exactly:

```text
year
income
```

`income` is expressed in constant 2025 U.S. dollars. The generated observations reproduce fitted distributions and annual statistical constraints; they are not recovered PNAD microdata.
