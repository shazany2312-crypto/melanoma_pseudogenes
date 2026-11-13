from pathlib import Path
import csv
import gzip
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
input_path = (
    PROJECT_DIR / "data" / "raw"
    / "TCGA_SKCM_sampleMap_HiSeqV2.gz"
)
output_dir = PROJECT_DIR / "results" / "input_check"
output_dir.mkdir(parents=True, exist_ok=True)

if not input_path.is_file():
    raise FileNotFoundError(f"Не найден файл: {input_path}")

with gzip.open(input_path, "rt", encoding="utf-8-sig") as file:
    header = next(csv.reader(file, delimiter="\t"))

sample_ids = header[1:]
if len(set(sample_ids)) != len(sample_ids):
    raise ValueError("В заголовке повторяются ID образцов.")

gene_reports = []
sample_missing = np.zeros(len(sample_ids), dtype=int)
sample_invalid = np.zeros(len(sample_ids), dtype=int)
sample_infinite = np.zeros(len(sample_ids), dtype=int)
sample_zero = np.zeros(len(sample_ids), dtype=int)
sample_min = np.full(len(sample_ids), np.inf)
sample_max = np.full(len(sample_ids), -np.inf)

missing_tokens = {"", "na", "nan", "n/a", "null", "none"}

reader = pd.read_csv(
    input_path,
    sep="\t",
    compression="gzip",
    dtype=str,
    keep_default_na=False,
    chunksize=1000
)

print("Проверяем всю матрицу...")

for chunk in reader:
    names = chunk.iloc[:, 0].str.strip()
    text = chunk.iloc[:, 1:].apply(lambda column: column.str.strip())

    missing = text.apply(
        lambda column: column.str.lower().isin(missing_tokens)
    ).to_numpy()

    numeric = text.apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)

    invalid = np.isnan(values) & ~missing
    infinite = np.isinf(values)
    finite = np.isfinite(values)
    zeros = finite & (values == 0)

    sample_missing += missing.sum(axis=0)
    sample_invalid += invalid.sum(axis=0)
    sample_infinite += infinite.sum(axis=0)
    sample_zero += zeros.sum(axis=0)

    low = np.where(finite, values, np.inf)
    high = np.where(finite, values, -np.inf)

    sample_min = np.minimum(sample_min, low.min(axis=0))
    sample_max = np.maximum(sample_max, high.max(axis=0))

    row_min = low.min(axis=1)
    row_max = high.max(axis=1)
    complete = finite.all(axis=1)

    report = pd.DataFrame({
        "gene_name": names.to_numpy(),
        "n_missing": missing.sum(axis=1),
        "n_non_numeric": invalid.sum(axis=1),
        "n_infinite": infinite.sum(axis=1),
        "n_zero": zeros.sum(axis=1),
        "all_zero": zeros.all(axis=1),
        "constant_complete_row": complete & (row_min == row_max),
        "min_finite": np.where(np.isfinite(row_min), row_min, np.nan),
        "max_finite": np.where(np.isfinite(row_max), row_max, np.nan)
    })
    gene_reports.append(report)

genes = pd.concat(gene_reports, ignore_index=True)
genes.insert(0, "data_row_number", np.arange(1, len(genes) + 1))
genes["duplicated_gene_name"] = genes["gene_name"].duplicated(keep=False)
genes["empty_gene_name"] = genes["gene_name"].eq("")

samples = pd.DataFrame({
    "sample_id": sample_ids,
    "n_missing": sample_missing,
    "n_non_numeric": sample_invalid,
    "n_infinite": sample_infinite,
    "n_zero": sample_zero,
    "min_finite": np.where(np.isfinite(sample_min), sample_min, np.nan),
    "max_finite": np.where(np.isfinite(sample_max), sample_max, np.nan)
})

summary = pd.DataFrame([
    ("n_data_rows", len(genes)),
    ("n_samples", len(samples)),
    ("n_unique_gene_names", genes["gene_name"].nunique()),
    ("n_empty_gene_names", int(genes["empty_gene_name"].sum())),
    ("n_rows_with_duplicated_gene_name",
     int(genes["duplicated_gene_name"].sum())),
    ("n_missing_cells", int(sample_missing.sum())),
    ("n_non_numeric_cells", int(sample_invalid.sum())),
    ("n_infinite_cells", int(sample_infinite.sum())),
    ("n_all_zero_rows", int(genes["all_zero"].sum())),
    ("n_constant_complete_rows",
     int(genes["constant_complete_row"].sum()))
], columns=["field", "value"])

flagged = (
    genes["duplicated_gene_name"]
    | genes["empty_gene_name"]
    | genes["all_zero"]
    | genes["constant_complete_row"]
    | (genes[["n_missing", "n_non_numeric", "n_infinite"]].sum(axis=1) > 0)
)

summary.to_csv(
    output_dir / "quality_summary.csv",
    index=False, encoding="utf-8-sig"
)
samples.to_csv(
    output_dir / "sample_quality.csv",
    index=False, encoding="utf-8-sig"
)
genes.to_csv(
    output_dir / "gene_quality.csv",
    index=False, encoding="utf-8-sig"
)
genes.loc[flagged].to_csv(
    output_dir / "flagged_rows.csv",
    index=False, encoding="utf-8-sig"
)

print(summary.to_string(index=False))
print(f"\nОтчёты сохранены в: {output_dir}")
print("Готово. Исходная матрица не изменена.")