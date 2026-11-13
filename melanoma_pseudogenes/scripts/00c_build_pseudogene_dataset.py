from pathlib import Path
import gzip
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
raw_dir = PROJECT_DIR / "data" / "raw"

expression_path = raw_dir / "TCGA_SKCM_sampleMap_HiSeqV2.gz"
gtf_path = raw_dir / "gencode.v50.annotation.gtf.gz"

output_dir = PROJECT_DIR / "results" / "dataset_build"
output_dir.mkdir(parents=True, exist_ok=True)

for path in [expression_path, gtf_path]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")

print("1. Читаем аннотацию GENCODE...")

records = []

with gzip.open(gtf_path, "rt", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")

        if len(fields) != 9 or fields[2] != "gene":
            continue

        attributes = dict(
            re.findall(r'(\S+)\s+"([^"]*)";', fields[8])
        )

        records.append({
            "gene_id": attributes.get("gene_id", ""),
            "gene_name": attributes.get("gene_name", ""),
            "gene_type": attributes.get("gene_type", ""),
            "chromosome": fields[0],
            "start": int(fields[3]),
            "end": int(fields[4]),
            "strand": fields[6]
        })

annotation = pd.DataFrame(records)

biotypes = (
    annotation["gene_type"]
    .value_counts()
    .rename_axis("gene_type")
    .reset_index(name="n_genes")
)

biotypes.to_csv(
    output_dir / "gencode_biotype_counts.csv",
    index=False,
    encoding="utf-8-sig"
)

selected = annotation.loc[
    annotation["gene_type"].eq("processed_pseudogene")
].copy()

if selected.empty:
    raise ValueError("В аннотации не найден биотип processed_pseudogene.")

if selected["gene_name"].eq("").any():
    raise ValueError("У части выбранных псевдогенов отсутствует gene_name.")

duplicate_mask = selected["gene_name"].duplicated(keep=False)

duplicates = selected.loc[duplicate_mask].copy()
duplicates["exclusion_reason"] = "ambiguous_gene_name_mapping"

duplicates.to_csv(
    output_dir / "annotation_duplicate_names.csv",
    index=False,
    encoding="utf-8-sig"
)

n_before_exclusion = len(selected)
n_ambiguous_names = duplicates["gene_name"].nunique()

selected = selected.loc[~duplicate_mask].copy()

print("Записей выбранного биотипа до исключения:", n_before_exclusion)
print("Неоднозначных названий:", n_ambiguous_names)
print("Исключено записей аннотации:", len(duplicates))
print("Осталось однозначных записей:", len(selected))

print("2. Загружаем общую матрицу экспрессии...")

expression = pd.read_csv(
    expression_path,
    sep="\t",
    compression="gzip",
    index_col=0
)

expression.index = expression.index.astype(str).str.strip()

if expression.index.duplicated().any():
    raise ValueError("В матрице повторяются названия строк.")

selected["found_in_expression"] = selected["gene_name"].isin(
    expression.index
)

selected.to_csv(
    output_dir / "processed_pseudogene_annotation.csv",
    index=False,
    encoding="utf-8-sig"
)

found_names = selected.loc[
    selected["found_in_expression"], "gene_name"
].tolist()

if not found_names:
    raise ValueError("Ни один псевдоген не сопоставился с матрицей.")

pseudogenes = expression.loc[found_names].copy()
pseudogenes.index.name = "gene_name"

pseudogenes.to_csv(
    output_dir / "processed_pseudogene_expression_unfiltered.csv",
    encoding="utf-8-sig"
)
summary = pd.DataFrame([
    ("selection_biotype", "processed_pseudogene"),
    ("matching_method", "exact_gene_name_after_whitespace_trim"),
    ("n_selected_in_gencode_before_exclusion", n_before_exclusion),
    ("n_ambiguous_names_excluded", n_ambiguous_names),
    ("n_annotation_records_excluded", len(duplicates)),
    ("n_unambiguous_selected_in_gencode", len(selected)),
    ("n_found_in_expression", len(found_names)),
    ("n_not_found_in_expression",
     int((~selected["found_in_expression"]).sum())),
    ("n_samples", pseudogenes.shape[1]),
    ("n_all_zero_pseudogenes",
     int(pseudogenes.eq(0).all(axis=1).sum())),
    ("expression_filter_applied", False),
    ("sample_filter_applied", False),
    ("additional_normalization_applied", False)
], columns=["field", "value"])

summary.to_csv(
    output_dir / "pseudogene_dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nБиотипы, содержащие processed_pseudogene:")
print(
    biotypes.loc[
        biotypes["gene_type"].str.contains(
            "processed_pseudogene", regex=False
        )
    ].to_string(index=False)
)

print("\nРезультат формирования датасета:")
print(summary.to_string(index=False))
print(f"\nФайлы сохранены в: {output_dir}")
print("Готово. Исходные файлы не изменены.")