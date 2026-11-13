from pathlib import Path
from itertools import combinations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def find_root():
    here=Path(__file__).resolve().parent
    for root in [here,here.parent]:
        if (root/"results"/"primary_metastatic_patient_sensitivity"/"included_samples.csv").is_file():
            return root
    raise FileNotFoundError("Save in project root or scripts folder, or specify --root")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path)
    parser.add_argument("--expression",type=Path)
    parser.add_argument("--annotation",type=Path)
    parser.add_argument("--outdir",type=Path)
    args=parser.parse_args()
    root=args.root.resolve() if args.root else find_root()
    source=root/"results"/"primary_metastatic_patient_sensitivity"
    ep=args.expression if args.expression else source/"patient_independent_expression.tsv.gz"
    ap=args.annotation if args.annotation else source/"included_samples.csv"
    expr=pd.read_csv(ep,sep="\t",index_col=0)
    ann=pd.read_csv(ap,dtype=str)
    if ann.sample_id.duplicated().any() or ann.patient_id.duplicated().any() or expr.index.has_duplicates or expr.columns.has_duplicates:
        raise ValueError("Duplicate sample, patient or gene IDs")
    if set(expr.columns)!=set(ann.sample_id):raise ValueError("Sample sets do not match")
    ann["sample_type_code"]=ann.sample_type_code.str.zfill(2)
    if not ann.sample_type_code.isin(["01","06","07"]).all():raise ValueError("Unexpected sample type")
    genes=["ANXA2P1","ANXA2P2","ANXA2P3"]
    if not set(genes).issubset(expr.index):raise ValueError("Missing ANXA2 pseudogene")
    rows=[]
    for group,codes in [("all",["01","06","07"]),("primary",["01"]),("metastatic",["06","07"])]:
        ids=ann.loc[ann.sample_type_code.isin(codes),"sample_id"]
        for first,second in combinations(genes,2):
            x=expr.loc[first,ids].to_numpy(dtype=float)
            y=expr.loc[second,ids].to_numpy(dtype=float)
            if len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all() or np.ptp(x)==0 or np.ptp(y)==0:
                raise ValueError(f"Invalid values for {group}/{first}/{second}")
            rho,p=spearmanr(x,y)
            rows.append(dict(group=group,gene_1=first,gene_2=second,n=len(x),spearman_rho=float(rho),pvalue=float(p)))
    results=pd.DataFrame(rows)
    mask=results.group.ne("all")
    results["BH_adjusted_p_six_within_group_tests"]=np.nan
    results.loc[mask,"BH_adjusted_p_six_within_group_tests"]=multipletests(results.loc[mask,"pvalue"],method="fdr_bh")[1]
    display=results.pivot(index=["gene_1","gene_2"],columns="group",values="spearman_rho").reset_index()
    dest=args.outdir if args.outdir else root/"results"/"anxa2p_within_group_correlations"
    dest.mkdir(parents=True,exist_ok=True)
    results.to_csv(dest/"anxa2p_correlations_by_group.csv",index=False,encoding="utf-8-sig")
    display.to_csv(dest/"anxa2p_rho_comparison.csv",index=False,encoding="utf-8-sig")
    notes="""Exploratory follow-up of the ANXA2 pseudogene branch observed in pooled clustering.
Same patient-independent sample set; no additional log transform, refiltering or sample exclusion.
Three pairwise Spearman correlations in each of two groups: six within-group tests.
BH is applied jointly to these six p-values only. Pooled correlations are descriptive references; pooled adjusted p-values are left blank.
This correction does not account for prior inspection of the pooled matrix or selection of this gene family.
P-values are SciPy asymptotic Spearman p-values; no permutation analysis is performed.
No test of differences between group-specific correlation coefficients is performed.
Within-group associations do not establish causality, ceRNA activity, or absence of technical/tissue-composition confounding.
Original outputs unchanged.
"""
    (dest/"README.txt").write_text(notes,encoding="utf-8")
    print("Samples: all =",len(ann),"; primary =",int(ann.sample_type_code.eq("01").sum()),"; metastatic =",int(ann.sample_type_code.isin(["06","07"]).sum()))
    print("Within-group tests:",int(mask.sum()))
    print("BH correction family: 6 within-group tests; exploratory post-selection analysis.")
    print(display.to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("Output:",dest.resolve())
    print("Original results unchanged.")

if __name__=="__main__":
    main()
