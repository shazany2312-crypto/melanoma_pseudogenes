from pathlib import Path
import re
import numpy as np
import pandas as pd

relative_pseudo = (
    Path("results") / "dataset_build"
    / "processed_pseudogene_expression_unfiltered.csv"
)
relative_mirna = (
    Path("data") / "raw"
    / "TCGA_SKCM_sampleMap_miRNA_HiSeq_gene.gz"
)

PROJECT_DIR = next(
    (
        folder
        for folder in Path(__file__).resolve().parents
        if (folder / relative_pseudo).is_file()
        and (folder / relative_mirna).is_file()
    ),
    None
)

if PROJECT_DIR is None:
    raise FileNotFoundError(
        "Не найден корень проекта с обоими входными файлами. "
        "Проверь расположение новой CSV-матрицы и исходного miRNA-файла."
    )

output_dir = PROJECT_DIR / "results" / "anxa2p1_mir186_check"
output_dir.mkdir(parents=True, exist_ok=True)

print("Корень проекта:", PROJECT_DIR)
print("1. Читаем матрицы...")

pseudo = pd.read_csv(
    PROJECT_DIR / relative_pseudo,
    index_col=0
)
mirna = pd.read_csv(
    PROJECT_DIR / relative_mirna,
    sep="\t",
    compression="gzip",
    index_col=0
)

pseudo.index = pseudo.index.astype(str).str.strip()
mirna.index = mirna.index.astype(str).str.strip()

for label, table in [("псевдогенов", pseudo), ("микроРНК", mirna)]:
    if table.index.duplicated().any():
        raise ValueError(f"В матрице {label} повторяются ID строк.")
    if table.columns.duplicated().any():
        raise ValueError(f"В матрице {label} повторяются ID образцов.")

if "ANXA2P1" not in pseudo.index:
    raise ValueError("ANXA2P1 отсутствует в матрице псевдогенов.")

if "MIMAT0000456" not in mirna.index:
    raise ValueError(
        "MIMAT0000456 отсутствует в miRNA-матрице. "
        "Не подбирай другой ID вручную: сначала проверим файл."
    )

common = sorted(set(pseudo.columns) & set(mirna.columns))

if not common:
    raise ValueError("Общих ID образцов не найдено.")

print("2. Сопоставляем одинаковые ID образцов...")

dataset = pd.DataFrame({"sample_id": common})

parts = dataset["sample_id"].str.extract(
    r"^(TCGA-[^-]+-[^-]+)-(\d{2})[A-Za-z]?$"
)

if parts.isna().any().any():
    raise ValueError(
        "Есть ID неожиданного формата. "
        "Нужно проверить их до разбора типов образцов."
    )

dataset["patient_id"] = parts[0]
dataset["sample_type_code"] = parts[1]

type_names = {
    "01": "Primary tumor",
    "06": "Metastatic tumor",
    "07": "Additional metastatic tumor",
    "11": "Solid tissue normal"
}

dataset["sample_type"] = (
    dataset["sample_type_code"].map(type_names).fillna("Other")
)

dataset["ANXA2P1_expression"] = pd.to_numeric(
    pseudo.loc["ANXA2P1", common], errors="raise"
).to_numpy()

dataset["miR186_5p_expression"] = pd.to_numeric(
    mirna.loc["MIMAT0000456", common], errors="raise"
).to_numpy()

values = dataset[
    ["ANXA2P1_expression", "miR186_5p_expression"]
].to_numpy(dtype=float)

dataset["both_values_finite"] = np.isfinite(values).all(axis=1)
dataset["is_tumor"] = dataset["sample_type_code"].isin(["01", "06", "07"])

dataset["patient_has_multiple_common_samples"] = (
    dataset["patient_id"].duplicated(keep=False)
)

dataset.to_csv(
    output_dir / "anxa2p1_mir186_matched_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

repeated = dataset.loc[
    dataset["patient_has_multiple_common_samples"]
].sort_values(["patient_id", "sample_id"])

repeated.to_csv(
    output_dir / "repeated_patient_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

sample_sets = {
    "pseudogene_matrix": set(pseudo.columns),
    "mirna_matrix": set(mirna.columns)
}

unmatched = []
for source, samples in sample_sets.items():
    for sample in sorted(samples - set(common)):
        unmatched.append({"source": source, "sample_id": sample})

pd.DataFrame(
    unmatched, columns=["source", "sample_id"]
).to_csv(
    output_dir / "unmatched_samples.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nОбщих образцов:", len(dataset))
print("Уникальных пациентов:", dataset["patient_id"].nunique())
print("\nТипы образцов:")
print(dataset["sample_type"].value_counts().to_string())
print("\nОбразцов с пропуском или бесконечностью:",
      int((~dataset["both_values_finite"]).sum()))
print("Образцов повторно представленных пациентов:", len(repeated))
print("Только в матрице псевдогенов:",
      len(sample_sets["pseudogene_matrix"] - set(common)))
print("Только в матрице микроРНК:",
      len(sample_sets["mirna_matrix"] - set(common)))

print(f"\nCSV сохранены в: {output_dir}")
print("Готово. Нормализация и исключение образцов не выполнялись.")