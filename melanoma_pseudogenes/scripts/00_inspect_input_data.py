from pathlib import Path
import csv
import gzip
import hashlib

PROJECT_DIR = Path(__file__).resolve().parents[1]
input_path = (
    PROJECT_DIR / "data" / "raw"
    / "TCGA_SKCM_sampleMap_HiSeqV2.gz"
)
output_dir = PROJECT_DIR / "results" / "input_check"
output_dir.mkdir(parents=True, exist_ok=True)

if not input_path.is_file():
    raise FileNotFoundError(f"Не найден файл: {input_path}")

print("Читаем файл. Дождись окончания работы...")

sha256 = hashlib.sha256()
with input_path.open("rb") as file:
    for block in iter(lambda: file.read(1024 * 1024), b""):
        sha256.update(block)

preview = []
gene_count = 0
wrong_width_count = 0

with gzip.open(input_path, "rt", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    header = next(reader)

    for row in reader:
        if not row:
            continue

        gene_count += 1

        if len(row) != len(header):
            wrong_width_count += 1

        if len(preview) < 20:
            preview.append(row)

with (output_dir / "expression_preview.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(preview)

with (output_dir / "sample_ids.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["sample_id"])
    writer.writerows([[sample] for sample in header[1:]])

summary = [
    ["input_filename", input_path.name],
    ["file_size_bytes", input_path.stat().st_size],
    ["sha256", sha256.hexdigest()],
    ["first_column_name", header[0]],
    ["n_data_rows", gene_count],
    ["n_sample_columns", len(header) - 1],
    ["n_rows_with_unexpected_column_count", wrong_width_count],
    ["normalization_status", "not_verified"],
    ["expression_scale_status", "not_verified"],
]

with (output_dir / "input_summary.csv").open(
    "w", encoding="utf-8-sig", newline=""
) as file:
    writer = csv.writer(file)
    writer.writerow(["field", "value"])
    writer.writerows(summary)

print(f"Строк данных: {gene_count}")
print(f"Столбцов образцов: {len(header) - 1}")
print(f"Строк с неожиданным числом столбцов: {wrong_width_count}")
print(f"CSV-файлы сохранены в: {output_dir}")
print("Готово.")