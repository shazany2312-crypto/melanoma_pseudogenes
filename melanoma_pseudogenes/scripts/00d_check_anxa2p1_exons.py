from pathlib import Path
import gzip
import re
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
raw_dir = PROJECT_DIR / "data" / "raw"
gtf_path = raw_dir / "gencode.v50.annotation.gtf.gz"
fasta_path = raw_dir / "anxa2p_pseudogenes_GRCh38.fasta"

output_dir = PROJECT_DIR / "results" / "anxa2p1_sequence_check"
output_dir.mkdir(parents=True, exist_ok=True)

for path in [gtf_path, fasta_path]:
    if not path.is_file():
        raise FileNotFoundError(f"Не найден файл: {path}")

sequences = {}
headers = {}

with fasta_path.open(encoding="utf-8-sig") as file:
    for line in file:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            name = line[1:].split()[0]
            headers[name] = line
            sequences[name] = ""
        else:
            sequences[name] += line.upper()

name = "ANXA2P1"
sequence = sequences[name]
header = headers[name]

region = re.search(r"\|\s*([^|\s]+):(\d+)\.\.(\d+):(-?1)", header)
if region is None:
    raise ValueError("Не удалось прочитать координаты из заголовка FASTA.")

chromosome = region.group(1)
chromosome = chromosome if chromosome.startswith("chr") else "chr" + chromosome
region_start = int(region.group(2))
region_end = int(region.group(3))
strand = "+" if region.group(4) == "1" else "-"

if len(sequence) != region_end - region_start + 1:
    raise ValueError("Длина FASTA не соответствует координатам региона.")

motif = "ATTCTTT"
positions = [
    match.start() + 1
    for match in re.finditer(f"(?={motif})", sequence)
]

if not positions:
    raise ValueError("Мотив ATTCTTT не найден.")

print("1. Извлекаем записи ANXA2P1 из GENCODE...")

records = []

with gzip.open(gtf_path, "rt", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#"):
            continue

        fields = line.rstrip("\n").split("\t")
        if len(fields) != 9:
            continue

        attributes = dict(re.findall(r'(\S+)\s+"([^"]*)";', fields[8]))
        if attributes.get("gene_name") != name:
            continue
        if fields[2] not in {"gene", "transcript", "exon"}:
            continue

        records.append({
            "gene_id": attributes.get("gene_id", ""),
            "gene_name": name,
            "transcript_id": attributes.get("transcript_id", ""),
            "feature": fields[2],
            "chromosome": fields[0],
            "start": int(fields[3]),
            "end": int(fields[4]),
            "strand": fields[6],
            "exon_number": attributes.get("exon_number", ""),
            "gene_type": attributes.get("gene_type", ""),
            "transcript_type": attributes.get("transcript_type", "")
        })

if not records:
    raise ValueError("ANXA2P1 не найден в аннотации.")

annotation = pd.DataFrame(records)
annotation.to_csv(
    output_dir / "anxa2p1_annotation_features.csv",
    index=False,
    encoding="utf-8-sig"
)

compatible = annotation.loc[
    annotation["chromosome"].eq(chromosome)
    & annotation["strand"].eq(strand)
]
exons = compatible.loc[compatible["feature"].eq("exon")]

transcript_ids = sorted(
    compatible.loc[
        compatible["transcript_id"].ne(""), "transcript_id"
    ].unique()
)

results = []

for position in positions:
    end_position = position + len(motif) - 1

    if strand == "+":
        genomic_start = region_start + position - 1
        genomic_end = region_start + end_position - 1
    else:
        genomic_start = region_end - end_position + 1
        genomic_end = region_end - position + 1

    for transcript_id in transcript_ids:
        tx_exons = exons.loc[exons["transcript_id"].eq(transcript_id)]

        covered = sum(
            any(
                row.start <= base <= row.end
                for row in tx_exons.itertuples()
            )
            for base in range(genomic_start, genomic_end + 1)
        )

        results.append({
            "gene_name": name,
            "transcript_id": transcript_id,
            "motif": motif,
            "fasta_start_1based": position,
            "fasta_end_1based": end_position,
            "chromosome": chromosome,
            "genomic_start_1based": genomic_start,
            "genomic_end_1based": genomic_end,
            "strand": strand,
            "n_exons": len(tx_exons),
            "n_motif_bases_in_exons": covered,
            "motif_fully_exonic": covered == len(motif)
        })

columns = [
    "gene_name", "transcript_id", "motif",
    "fasta_start_1based", "fasta_end_1based", "chromosome",
    "genomic_start_1based", "genomic_end_1based", "strand",
    "n_exons", "n_motif_bases_in_exons", "motif_fully_exonic"
]

report = pd.DataFrame(results, columns=columns)
report.to_csv(
    output_dir / "anxa2p1_motif_exon_check.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nНайденные записи:")
print(annotation["feature"].value_counts().to_string())
print("\nПроверка мотива:")
if report.empty:
    print("Совместимые записи транскриптов не найдены.")
else:
    print(report.to_string(index=False))

print(f"\nОтчёты сохранены в: {output_dir}")
print("Готово. Исходные файлы не изменены.")