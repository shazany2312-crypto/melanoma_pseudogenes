from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

relative_dir = Path("results") / "mir186_target_dataset"

PROJECT_DIR = next(
    (
        folder
        for folder in Path(__file__).resolve().parents
        if (folder / relative_dir / "target_expression_unfiltered.csv").is_file()
        and (folder / relative_dir / "analysis_sample_metadata.csv").is_file()
    ),
    None
)

if PROJECT_DIR is None:
    raise FileNotFoundError("Не найдены матрица мишеней и метаданные.")

input_dir = PROJECT_DIR / relative_dir
output_dir = PROJECT_DIR / "results" / "mir186_target_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

expression = pd.read_csv(
    input_dir / "target_expression_unfiltered.csv",
    index_col=0
)
metadata = pd.read_csv(
    input_dir / "analysis_sample_metadata.csv",
    dtype={"sample_type_code": str}
)

metadata["sample_type_code"] = (
    metadata["sample_type_code"].str.zfill(2)
)

if metadata["sample_id"].duplicated().any():
    raise ValueError("В метаданных повторяются образцы.")

if metadata["patient_id"].duplicated().any():
    raise ValueError("В метаданных повторяются пациенты.")

if expression.index.duplicated().any():
    raise ValueError("Повторяются названия мишеней.")

if expression.columns.duplicated().any():
    raise ValueError("Повторяются столбцы образцов.")

if not metadata["sample_type_code"].isin(["01", "06"]).all():
    raise ValueError("В основном наборе есть другие типы образцов.")

metadata = metadata.set_index("sample_id")
sample_ids = metadata.index.tolist()

if set(expression.columns) != set(sample_ids):
    raise ValueError("Наборы образцов матрицы и метаданных различаются.")

expression = expression.loc[:, sample_ids].apply(
    pd.to_numeric, errors="raise"
)

reference_columns = ["ANXA2P1_expression", "miR186_5p_expression"]
metadata[reference_columns] = metadata[reference_columns].apply(
    pd.to_numeric, errors="raise"
)

if not np.isfinite(metadata[reference_columns].to_numpy()).all():
    raise ValueError("В измерениях ANXA2P1 или miR-186-5p есть пропуски.")

primary_ids = metadata.index[
    metadata["sample_type_code"].eq("01")
].tolist()
metastatic_ids = metadata.index[
    metadata["sample_type_code"].eq("06")
].tolist()

if not primary_ids or not metastatic_ids:
    raise ValueError("Одна из групп пуста.")

MINIMUM_FRACTION = 0.10
minimum_primary = math.ceil(len(primary_ids) * MINIMUM_FRACTION)
minimum_metastatic = math.ceil(len(metastatic_ids) * MINIMUM_FRACTION)

quality = pd.DataFrame(index=expression.index)
quality.index.name = "target_gene"

quality["all_values_finite"] = np.isfinite(
    expression.to_numpy()
).all(axis=1)

quality["n_nonzero_primary"] = expression[primary_ids].ne(0).sum(axis=1)
quality["n_nonzero_metastatic"] = (
    expression[metastatic_ids].ne(0).sum(axis=1)
)

quality["minimum_nonzero_primary"] = minimum_primary
quality["minimum_nonzero_metastatic"] = minimum_metastatic

quality["included"] = (
    quality["all_values_finite"]
    & (
        (quality["n_nonzero_primary"] >= minimum_primary)
        | (quality["n_nonzero_metastatic"] >= minimum_metastatic)
    )
)

quality.to_csv(
    output_dir / "target_inclusion_report.csv",
    encoding="utf-8-sig"
)

selected = expression.loc[quality["included"]].copy()

if selected.empty:
    raise ValueError("После фильтрации не осталось генов.")

selected.to_csv(
    output_dir / "target_expression_filtered.csv",
    encoding="utf-8-sig"
)

groups = {
    "all_main": sample_ids,
    "primary": primary_ids,
    "metastatic": metastatic_ids
}

references = {
    "mirna": "miR186_5p_expression",
    "pseudogene": "ANXA2P1_expression"
}

rows = []

print("Рассчитываем корреляции...")

for group, ids in groups.items():
    for gene in selected.index:
        y = selected.loc[gene, ids].to_numpy(dtype=float)

        for relation, column in references.items():
            x = metadata.loc[ids, column].to_numpy(dtype=float)

            if len(ids) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
                rho, pvalue = np.nan, np.nan
                status = "insufficient_or_constant_data"
            else:
                rho, pvalue = spearmanr(x, y, alternative="two-sided")
                status = "ok"

            rows.append({
                "target_gene": gene,
                "group": group,
                "relation": relation,
                "n_samples": len(ids),
                "spearman_rho": rho,
                "pvalue": pvalue,
                "status": status
            })

results = pd.DataFrame(rows)
results["FDR_BH_all_current_tests"] = np.nan
valid = results["pvalue"].notna()

if valid.any():
    results.loc[valid, "FDR_BH_all_current_tests"] = multipletests(
        results.loc[valid, "pvalue"],
        method="fdr_bh"
    )[1]

results.to_csv(
    output_dir / "all_target_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)

mirna = results.loc[results["relation"].eq("mirna")].copy()
pseudo = results.loc[results["relation"].eq("pseudogene")].copy()

paired = mirna.merge(
    pseudo,
    on=["target_gene", "group"],
    suffixes=("_mirna", "_pseudogene"),
    validate="one_to_one"
)

paired["candidate_pattern"] = (
    (paired["spearman_rho_mirna"] < 0)
    & (paired["spearman_rho_pseudogene"] > 0)
    & (paired["FDR_BH_all_current_tests_mirna"] < 0.05)
    & (paired["FDR_BH_all_current_tests_pseudogene"] < 0.05)
)

paired.to_csv(
    output_dir / "paired_target_results.csv",
    index=False,
    encoding="utf-8-sig"
)

paired.loc[paired["candidate_pattern"]].to_csv(
    output_dir / "candidate_patterns.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Генов до фильтрации:", len(expression))
print("Генов после фильтрации:", len(selected))
print("Минимум ненулевых значений в первичных:", minimum_primary)
print("Минимум ненулевых значений в метастазах:", minimum_metastatic)
print("Успешно рассчитанных тестов:", int(valid.sum()))

print("\nЧисло кандидатных паттернов по группам:")
print(
    paired.groupby("group")["candidate_pattern"]
    .sum()
    .reindex(groups.keys())
    .to_string()
)

print(f"\nРезультаты сохранены в: {output_dir}")
print("Готово. Анализ разведочный; предшествующий поиск miRNA не учтён FDR.")