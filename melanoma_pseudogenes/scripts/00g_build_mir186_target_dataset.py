from pathlib import Path
import pandas as pd

relative_targetscan = (
    Path("data") / "raw"
    / "TargetScan8.0__miR-186-5p.Human.predicted_targets.txt"
)
relative_expression = (
    Path("data") / "raw"
    / "TCGA_SKCM_sampleMap_HiSeqV2.gz"
)
relative_samples = (
    Path("results") / "anxa2p1_mir186_analysis"
    / "analysis_dataset.csv"
)

required_paths = [
    relative_targetscan,
    relative_expression,
    relative_samples
]

PROJECT_DIR = next(
    (
        folder
        for folder in Path(__file__).resolve().parents
        if all((folder / path).is_file() for path in required_paths)
    ),
    None
)

if PROJECT_DIR is None:
    raise FileNotFoundError(
        "Не найдены все три входных файла в одном проекте."
    )

output_dir = PROJECT_DIR / "results" / "mir186_target_dataset"
output_dir.mkdir(parents=True, exist_ok=True)

print("1. Проверяем список TargetScan...")

targets = pd.read_csv(
    PROJECT_DIR / relative_targetscan,
    sep="\t"
)

required_columns = [
    "Target gene",
    "Representative transcript",
    "Representative miRNA"
]

for column in required_columns:
    if column not in targets.columns:
        raise ValueError(f"Нет обязательного столбца: {column}")

targets["Target gene"] = (
    targets["Target gene"].astype("string").str.strip()
)

if targets["Target gene"].isna().any() or targets["Target gene"].eq("").any():
    raise ValueError("Есть пустые названия генов в TargetScan.")

if not targets["Representative miRNA"].eq("hsa-miR-186-5p").all():
    raise ValueError("В файле есть записи для другой микроРНК.")

targets.to_csv(
    output_dir / "targetscan_source_records.csv",
    index=False,
    encoding="utf-8-sig"
)

gene_list = (
    targets.groupby("Target gene", sort=True)
    .agg(
        n_source_records=("Representative transcript", "size"),
        representative_transcripts=(
            "Representative transcript",
            lambda values: ";".join(sorted(set(values.dropna())))
        )
    )
    .reset_index()
    .rename(columns={"Target gene": "target_gene"})
)

print("2. Читаем основной набор образцов...")

samples = pd.read_csv(
    PROJECT_DIR / relative_samples,
    dtype={"sample_type_code": str}
)

if samples["sample_id"].duplicated().any():
    raise ValueError("Повторяются ID образцов.")

if samples["patient_id"].duplicated().any():
    raise ValueError("В основном наборе повторяются пациенты.")

sample_ids = samples["sample_id"].tolist()

print("3. Читаем полную матрицу экспрессии...")

expression = pd.read_csv(
    PROJECT_DIR / relative_expression,
    sep="\t",
    compression="gzip",
    index_col=0
)

expression.index = expression.index.astype(str).str.strip()

if expression.index.duplicated().any():
    raise ValueError("В матрице повторяются названия генов.")

missing_samples = sorted(set(sample_ids) - set(expression.columns))
if missing_samples:
    raise ValueError(
        f"В матрице отсутствуют образцы: {missing_samples}"
    )

gene_list["found_in_expression"] = (
    gene_list["target_gene"].isin(expression.index)
)

gene_list.to_csv(
    output_dir / "target_gene_mapping.csv",
    index=False,
    encoding="utf-8-sig"
)

found_genes = gene_list.loc[
    gene_list["found_in_expression"], "target_gene"
].tolist()

if not found_genes:
    raise ValueError("Ни одна мишень не сопоставилась.")

target_expression = expression.loc[
    found_genes, sample_ids
].copy()

target_expression.index.name = "target_gene"

target_expression.to_csv(
    output_dir / "target_expression_unfiltered.csv",
    encoding="utf-8-sig"
)

samples.to_csv(
    output_dir / "analysis_sample_metadata.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = pd.DataFrame([
    ("n_targetscan_records", len(targets)),
    ("n_unique_target_symbols", len(gene_list)),
    ("n_found_in_expression", len(found_genes)),
    ("n_not_found_in_expression",
     int((~gene_list["found_in_expression"]).sum())),
    ("n_samples", len(sample_ids)),
    ("n_all_zero_targets",
     int(target_expression.eq(0).all(axis=1).sum())),
    ("expression_filter_applied", False),
    ("additional_normalization_applied", False)
], columns=["field", "value"])

summary.to_csv(
    output_dir / "target_dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nРезультат:")
print(summary.to_string(index=False))
print(f"\nCSV сохранены в: {output_dir}")
print("Готово. Корреляции ещё не рассчитывались.")