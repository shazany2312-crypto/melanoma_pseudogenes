from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list

def main():
    parser = argparse.ArgumentParser(description="Plot existing miR-186 candidate correlations")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    default = root / "results" / "mir186_target_analysis" / "paired_target_results.csv"
    source = args.input if args.input else default
    if not source.is_file() and args.input is None:
        source = root / "paired_target_results.csv"
    if not source.is_file():
        raise FileNotFoundError(f"Input CSV not found: {source}. Use --input to specify it.")
    dest = args.outdir if args.outdir else source.parent / "heatmap_visualization"
    dest.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    groups = ["all_main", "primary", "metastatic"]
    relations = ["mirna", "pseudogene"]
    required = {"target_gene", "group", "candidate_pattern"}
    for rel in relations:
        required.update([f"spearman_rho_{rel}", f"FDR_BH_all_current_tests_{rel}"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data.duplicated(["target_gene", "group"]).any():
        raise ValueError("Duplicate gene/group rows")
    flag = data["candidate_pattern"].astype(str).str.lower()
    if not flag.isin(["true", "false"]).all():
        raise ValueError("candidate_pattern must contain True/False")
    selected = data.loc[flag.eq("true")]
    genes = sorted(selected.target_gene.unique())
    if len(genes) < 2:
        raise ValueError("At least two candidates are required for clustering")
    values = pd.DataFrame(index=genes)
    adjusted = pd.DataFrame(index=genes)
    membership = pd.DataFrame(index=genes)
    for group in groups:
        sub = data.loc[data.group.eq(group)].set_index("target_gene")
        membership[group] = pd.Index(genes).isin(selected.loc[selected.group.eq(group), "target_gene"])
        for rel in relations:
            key = f"{group}__{rel}"
            values[key] = sub.loc[genes, f"spearman_rho_{rel}"]
            adjusted[key] = sub.loc[genes, f"FDR_BH_all_current_tests_{rel}"]
    if not np.isfinite(values.to_numpy()).all() or not np.isfinite(adjusted.to_numpy()).all():
        raise ValueError("Nonfinite correlation or adjusted p-value")
    order = leaves_list(linkage(values.to_numpy(), method="average", metric="euclidean"))
    values = values.iloc[order]
    adjusted = adjusted.loc[values.index]
    fig, ax = plt.subplots(figsize=(10, max(7, len(genes)*0.36+2.5)))
    image = ax.imshow(values.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    labels = ["All: miRNA", "All: ANXA2P1", "Primary: miRNA", "Primary: ANXA2P1", "Metastatic: miRNA", "Metastatic: ANXA2P1"]
    ax.set_xticks(range(6), labels, rotation=40, ha="right")
    ax.set_yticks(range(len(values)), values.index)
    for i in range(len(values)):
        for j in range(6):
            star = "*" if adjusted.iloc[i,j] < 0.05 else ""
            ax.text(j, i, f"{values.iloc[i,j]:.2f}{star}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Spearman rho", shrink=0.65)
    ax.set_title(f"{len(genes)} exploratory candidates: correlation profiles\nAverage-linkage clustering; Euclidean distance", fontsize=11)
    fig.text(0.05, 0.01, "* BH-adjusted p < 0.05 for an individual correlation.\nSelection and clustering use the same data; this is not independent validation.", fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(dest / "candidate_heatmap.jpg", dpi=180, facecolor="white", pil_kwargs={"quality":95})
    fig.savefig(dest / "candidate_heatmap.pdf", facecolor="white")
    plt.close(fig)
    values.to_csv(dest / "heatmap_correlations.csv", index_label="target_gene")
    adjusted.to_csv(dest / "heatmap_BH_adjusted_pvalues.csv", index_label="target_gene")
    membership.to_csv(dest / "candidate_group_membership.csv", index_label="target_gene")
    print(f"Input: {source.resolve()}")
    print(f"Unique candidates: {len(genes)}")
    for group in groups:
        print(f"{group}: {int(membership[group].sum())}")
    print(f"Primary AND metastatic: {int((membership.primary & membership.metastatic).sum())}")
    print(f"Output: {dest.resolve()}")
    print("Done. Existing correlation results were plotted, not recalculated.")

if __name__ == "__main__":
    main()
