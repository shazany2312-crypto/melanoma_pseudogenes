from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def locate(root,name):
    files=[p for p in root.rglob(name) if ".venv" not in p.parts and ".git" not in p.parts]
    if len(files)!=1:
        raise ValueError(f"Expected one {name}, found {len(files)}: {files}")
    return files[0]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--outdir",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    paths={n:locate(root,n) for n in ["anxa2p1_mir186_matched_samples.csv","target_expression_unfiltered.csv","paired_target_results.csv"]}
    meta=pd.read_csv(paths["anxa2p1_mir186_matched_samples.csv"])
    expr=pd.read_csv(paths["target_expression_unfiltered.csv"])
    saved=pd.read_csv(paths["paired_target_results.csv"])
    if meta.sample_id.duplicated().any() or expr.target_gene.duplicated().any() or saved.duplicated(["target_gene","group"]).any():
        raise ValueError("Duplicate identifiers")
    meta=meta.set_index("sample_id")
    expr=expr.set_index("target_gene")
    meta=meta.loc[expr.columns].copy()
    summary=[]
    detail=[]
    for gene in ["MDGA1","ZBTB7A","ZNF385A"]:
        for group,code in [("primary",1),("metastatic",6)]:
            subset=meta.loc[pd.to_numeric(meta.sample_type_code).eq(code)]
            ids=subset.index.to_numpy()
            y=expr.loc[gene,ids].to_numpy(dtype=float)
            for rel,column in [("mirna","miR186_5p_expression"),("pseudogene","ANXA2P1_expression")]:
                x=subset[column].to_numpy(dtype=float)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("Nonfinite input; no samples were silently removed")
                row=saved.loc[saved.target_gene.eq(gene)&saved.group.eq(group)]
                if len(row)!=1:
                    raise ValueError("Missing saved result")
                row=row.iloc[0]
                full=float(spearmanr(x,y).statistic)
                if len(x)!=int(row[f"n_samples_{rel}"]) or not np.isclose(full,row[f"spearman_rho_{rel}"],rtol=0,atol=1e-10):
                    raise ValueError(f"Baseline mismatch for {gene}/{group}/{rel}")
                loo=[]
                for i,sid in enumerate(ids):
                    keep=np.arange(len(x))!=i
                    rho=float(spearmanr(x[keep],y[keep]).statistic)
                    if not np.isfinite(rho):
                        raise ValueError("Undefined leave-one-out coefficient")
                    loo.append(rho)
                    detail.append(dict(target_gene=gene,group=group,relation=rel,omitted_sample_id=sid,omitted_patient_id=subset.loc[sid,"patient_id"],n_remaining=len(x)-1,rho_full=full,rho_without_sample=rho,delta_rho=rho-full))
                loo=np.asarray(loo)
                delta=np.abs(loo-full)
                most=int(np.argmax(delta))
                summary.append(dict(target_gene=gene,group=group,relation=rel,n_full=len(x),rho_full=full,rho_loo_min=loo.min(),rho_loo_max=loo.max(),max_abs_delta_rho=delta[most],n_sign_changes=int((np.sign(loo)!=np.sign(full)).sum()),most_influential_sample_id=ids[most],rho_without_most_influential=loo[most]))
    dest=args.outdir if args.outdir else paths["paired_target_results.csv"].parent/"leave_one_out_sensitivity"
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(dest/"loo_summary.csv",index=False)
    pd.DataFrame(detail).to_csv(dest/"loo_sample_details.csv",index=False)
    print("Baseline validation passed: 12 correlations match saved results.")
    print("Correlations checked:",len(summary))
    print("Leave-one-out recalculations:",len(detail))
    print("Correlations with any sign change:",sum(d["n_sign_changes"]>0 for d in summary))
    print(pd.DataFrame(summary)[["target_gene","group","relation","rho_full","rho_loo_min","rho_loo_max","max_abs_delta_rho"]].to_string(index=False,float_format=lambda x:f"{x:.4f}"))
    print("Output:",dest.resolve())
    print("Original data and BH-adjusted p-values unchanged. No new FDR claims are made.")

if __name__=="__main__":
    main()
