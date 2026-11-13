from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

relative_input = (
    Path("results") / "anxa2p1_mir186_check"
    / "anxa2p1_mir186_matched_samples.csv"
)

PROJECT_DIR = next(
    (
        folder
        for folder in Path(__file__).resolve().parents
        if (folder / relative_input).is_file()
    ),
    None
)

if PROJECT_DIR is None:
    raise FileNotFoundError("Не найдена согласованная таблица образцов.")

output_dir = PROJECT_DIR / "results" / "anxa2p1_mir186_analysis"
output_dir.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(
    PROJECT_DIR / relative_input,
    dtype={"sample_type_code": str}
)

data["sample_type_code"] = data["sample_type_code"].str.zfill(2)

if data["sample_id"].duplicated().any():
    raise ValueError("Повторяются ID образцов.")

value_columns = ["ANXA2P1_expression", "miR186_5p_expression"]
data[value_columns] = data[value_columns].apply(
    pd.to_numeric, errors="raise"
)

finite = np.isfinite(data[value_columns].to_numpy()).all(axis=1)
eligible = data["sample_type_code"].isin(["01", "06"]) & finite

eligible_data = data.loc[eligible].copy()
repeated_patients = eligible_data.loc[
    eligible_data["patient_id"].duplicated(keep=False),
    "patient_id"
].unique()

data["included_main"] = (
    eligible & ~data["patient_id"].isin(repeated_patients)
)

data["decision_reason"] = "included"
data.loc[
    data["patient_id"].isin(repeated_patients),
    "decision_reason"
] = "multiple_eligible_samples_for_patient"
data.loc[
    ~data["sample_type_code"].isin(["01", "06"]),
    "decision_reason"
] = "sample_type_not_01_or_06"
data.loc[
    ~finite,
    "decision_reason"
] = "non_finite_expression"

main = data.loc[data["included_main"]].copy()

if main["patient_id"].duplicated().any():
    raise ValueError("В основном наборе повторяются пациенты.")

if main.empty:
    raise ValueError("Основной набор пуст.")

main.to_csv(
    output_dir / "analysis_dataset.csv",
    index=False,
    encoding="utf-8-sig"
)
data.to_csv(
    output_dir / "sample_inclusion_audit.csv",
    index=False,
    encoding="utf-8-sig"
)

def calculate(label, table):
    if (
        len(table) < 3
        or table[value_columns[0]].nunique() < 2
        or table[value_columns[1]].nunique() < 2
    ):
        return {
            "analysis": label,
            "n_samples": len(table),
            "spearman_rho": np.nan,
            "pvalue": np.nan,
            "status": "insufficient_or_constant_data"
        }

    rho, pvalue = spearmanr(
        table[value_columns[0]],
        table[value_columns[1]],
        alternative="two-sided"
    )

    return {
        "analysis": label,
        "n_samples": len(table),
        "spearman_rho": rho,
        "pvalue": pvalue,
        "status": "ok"
    }

results = pd.DataFrame([
    calculate("all_main", main),
    calculate("primary_main", main.loc[
        main["sample_type_code"].eq("01")
    ]),
    calculate("metastatic_main", main.loc[
        main["sample_type_code"].eq("06")
    ])
])

results["FDR_BH_within_3_checks"] = np.nan
valid = results["pvalue"].notna()

if valid.any():
    results.loc[valid, "FDR_BH_within_3_checks"] = multipletests(
        results.loc[valid, "pvalue"],
        method="fdr_bh"
    )[1]

results["interpretation_scope"] = (
    "exploratory; does not adjust for prior miRNA candidate screening"
)

results.to_csv(
    output_dir / "main_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)

sensitivity_rows = []

for preferred_type in ["01", "06"]:
    alternative = eligible_data.copy()
    alternative["selection_priority"] = (
        ~alternative["sample_type_code"].eq(preferred_type)
    ).astype(int)

    alternative = (
        alternative
        .sort_values(["patient_id", "selection_priority", "sample_id"])
        .drop_duplicates("patient_id", keep="first")
    )

    label = f"one_sample_per_patient_prefer_{preferred_type}"
    sensitivity_rows.append(calculate(label, alternative))

    alternative.to_csv(
        output_dir / f"sensitivity_dataset_prefer_{preferred_type}.csv",
        index=False,
        encoding="utf-8-sig"
    )

pd.DataFrame(sensitivity_rows).to_csv(
    output_dir / "sensitivity_correlations.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Основной набор:")
print(main["sample_type"].value_counts().to_string())
print("Уникальных пациентов:", main["patient_id"].nunique())

print("\nОсновные корреляции:")
print(results.to_string(index=False))

print("\nПроверка устойчивости к выбору повторного образца:")
print(pd.DataFrame(sensitivity_rows).to_string(index=False))

print(f"\nРезультаты сохранены в: {output_dir}")
print("Готово. Исходные данные не изменены.")