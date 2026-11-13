from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

GENES = ["MDGA1", "ZBTB7A", "ZNF385A"]

def find_file(root, name):
    matches = [p for p in root.rglob(name) if not any(x in p.parts for x in [".venv", ".git"]) ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}; found {len(matches)}: {matches}")
    return matches[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--outdir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = {name:find_file(root,name) for name in ["anxa2p1_mir186_matched_samples.csv", "target_expression_unfiltered.csv", "paired_target_results.csv"]}
    samples = pd.read_csv(paths["anxa2p1_mir186_matched_samples.csv"])
    expression = pd.read_csv(paths["target_expression_unfiltered.csv"])
    results = pd.read_csv(paths["paired_target_results.csv"])
    if samples.sample_id.duplicated().any() or expression.target_gene.duplicated().any():
        raise ValueError("Duplicate sample IDs or target genes")
    if results.duplicated(["target_gene","group"]).any():
        raise ValueError("Duplicate gene/group result rows")
    samples = samples.set_index("sample_id")
    targets = expression.set_index("target_gene")
    ids = targets.columns.tolist()
    if not set(ids).issubset(samples.index):
        raise ValueError("Expression sample IDs missing from matched-sample table")
    aligned = samples.loc[ids].copy()
    groups = [("primary",1,"Primary tumors"),("metastatic",6,"Metastatic tumors")]
    relations = [("mirna","miR186_5p_expression","miR-186-5p"),("pseudogene","ANXA2P1_expression","ANXA2P1")]
    records = []
    plots = {}
    checks = []
    for gene in GENES:
        for group, sample_code, title in groups:
            subset = aligned.loc[pd.to_numeric(aligned.sample_type_code).eq(sample_code)]
            row = results.loc[results.target_gene.eq(gene) & results.group.eq(group)]
            if len(row) != 1:
                raise ValueError(f"Missing result for {gene}/{group}")
            row = row.iloc[0]
            for rel, column, label in relations:
                x = subset[column].to_numpy(dtype=float)
                y = targets.loc[gene,subset.index].to_numpy(dtype=float)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite values for {gene}/{group}/{rel}; stopping without dropping samples")
                rho = float(spearmanr(x,y).statistic)
                expected = float(row[f"spearman_rho_{rel}"])
                expected_n = int(row[f"n_samples_{rel}"])
                if len(x) != expected_n or not np.isclose(rho,expected,rtol=0,atol=1e-10):
                    raise ValueError(f"Mismatch {gene}/{group}/{rel}: n={len(x)} vs {expected_n}; rho={rho} vs {expected}")
                adjusted = float(row[f"FDR_BH_all_current_tests_{rel}"])
                plots[(gene,group,rel)] = (x,y,rho,adjusted,title,label)
                checks.append(dict(target_gene=gene,group=group,relation=rel,n=len(x),rho_recomputed=rho,rho_saved=expected,BH_adjusted_p=adjusted))
            for sid in subset.index:
                records.append(dict(target_gene=gene,group=group,sample_id=sid,patient_id=subset.loc[sid,"patient_id"],target_expression=targets.loc[gene,sid],ANXA2P1_expression=subset.loc[sid,"ANXA2P1_expression"],miR186_5p_expression=subset.loc[sid,"miR186_5p_expression"]))
    dest = args.outdir if args.outdir else paths["paired_target_results.csv"].parent / "candidate_scatterplots"
    dest.mkdir(parents=True,exist_ok=True)
    for gene in GENES:
        fig, axes = plt.subplots(2,2,figsize=(11,9),sharey=True)
        for col,(group,_,_) in enumerate(groups):
            for row,(rel,_,_) in enumerate(relations):
                x,y,rho,adjusted,title,label = plots[(gene,group,rel)]
                ax = axes[row,col]
                ax.scatter(x,y,s=22,alpha=0.55,color="#3268a8" if group=="primary" else "#b45c37",edgecolors="none")
                ax.set_title(f"{title} | n={len(x)}\nSpearman rho={rho:.3f}; BH-adjusted p={adjusted:.3g}",fontsize=10)
                ax.set_xlabel(f"{label}: input RNA expression value")
                ax.set_ylabel(f"{gene}: input RNA expression value")
                ax.grid(alpha=0.18)
        fig.suptitle(f"{gene}: exploratory expression associations",fontsize=14)
        fig.text(0.05,0.012,"Input expression scale retained; no new log transform or sample removal.\nAdjusted p-values copied from the original 5292-test analysis; not recalculated on this subset.",fontsize=9)
        fig.tight_layout(rect=[0,0.065,1,0.95])
        fig.savefig(dest/f"{gene}_scatterplots.jpg",dpi=180,facecolor="white",pil_kwargs={"quality":95})
        fig.savefig(dest/f"{gene}_scatterplots.pdf",facecolor="white")
        plt.close(fig)
    pd.DataFrame(records).to_csv(dest/"scatterplot_input_values.csv",index=False)
    pd.DataFrame(checks).to_csv(dest/"scatterplot_validation.csv",index=False)
    print("Validation passed: all 12 correlations and sample counts match saved results.")
    print("Primary samples per panel:",len(aligned.loc[pd.to_numeric(aligned.sample_type_code).eq(1)]))
    print("Metastatic samples per panel:",len(aligned.loc[pd.to_numeric(aligned.sample_type_code).eq(6)]))
    print("Created: 3 JPG, 3 PDF, and 2 CSV files.")
    print("Output:",dest.resolve())

if __name__ == "__main__":
    main()
