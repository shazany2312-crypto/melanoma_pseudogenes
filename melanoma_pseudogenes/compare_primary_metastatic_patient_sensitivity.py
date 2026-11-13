from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import scipy
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def find_root():
    here=Path(__file__).resolve().parent
    for candidate in [here,here.parent]:
        if (candidate/"results"/"tables"/"skcm_sample_annotation.tsv").is_file():
            return candidate
    raise FileNotFoundError("Project root not found. Save script in the project root or scripts folder, or use --root.")


def calculate(expr, annotation, genes):
    primary=annotation.loc[annotation.sample_type_code.eq("01"),"sample_id"].tolist()
    metastatic=annotation.loc[annotation.sample_type_code.isin(["06","07"]),"sample_id"].tolist()
    if not primary or not metastatic:
        raise ValueError("Both groups must contain samples")
    rows=[]
    for gene in genes:
        a=expr.loc[gene,primary].to_numpy(dtype=float)
        b=expr.loc[gene,metastatic].to_numpy(dtype=float)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError(f"Nonfinite expression for {gene}; no samples were silently dropped")
        u,p=mannwhitneyu(a,b,alternative="two-sided",method="auto")
        rows.append(dict(pseudogene=gene,n_primary=len(a),n_metastatic=len(b),mean_primary=a.mean(),mean_metastatic=b.mean(),median_primary=np.median(a),median_metastatic=np.median(b),mean_difference_metastatic_minus_primary=b.mean()-a.mean(),U_primary=float(u),pvalue=float(p)))
    result=pd.DataFrame(rows)
    result["FDR_BH"]=multipletests(result.pvalue,method="fdr_bh")[1]
    result["passes_FDR_0_05"]=result.FDR_BH.lt(.05)
    result["passes_FDR_and_abs_difference_0_5"]=result.passes_FDR_0_05 & result.mean_difference_metastatic_minus_primary.abs().gt(.5)
    return result


def main():
    parser=argparse.ArgumentParser(description="Patient-independent sensitivity comparison; preserve original 42-gene selection")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--expression",type=Path)
    parser.add_argument("--annotation",type=Path)
    parser.add_argument("--original",type=Path)
    parser.add_argument("--outdir",type=Path)
    args=parser.parse_args()
    root=args.root.resolve() if args.root else find_root()
    expression_path=args.expression if args.expression else root/"data"/"processed"/"skcm_tumor_expressed_processed_pseudogenes.tsv.gz"
    annotation_path=args.annotation if args.annotation else root/"results"/"tables"/"skcm_sample_annotation.tsv"
    original_path=args.original if args.original else root/"results"/"tables"/"primary_vs_metastatic_pseudogene_expression.tsv"
    for path in [expression_path,annotation_path,original_path]:
        if not path.is_file(): raise FileNotFoundError(str(path))
    expr=pd.read_csv(expression_path,sep="," if expression_path.suffix.lower()==".csv" else "\t",index_col=0)
    ann=pd.read_csv(annotation_path,sep="\t",dtype=str)
    original=pd.read_csv(original_path,sep="\t")
    if expr.index.has_duplicates or expr.columns.has_duplicates or ann.sample_id.duplicated().any() or original.pseudogene.duplicated().any():
        raise ValueError("Duplicate gene or sample identifiers")
    if not ann.sample_id.str.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}-[0-9]{2}").all():
        raise ValueError("Unexpected sample ID format; patient parsing needs review")
    ann["patient_id"]=ann.sample_id.str[:12]
    ann["sample_type_code"]=ann.sample_type_code.str.zfill(2)
    if not ann.sample_type_code.eq(ann.sample_id.str[-2:]).all():
        raise ValueError("Sample type code conflicts with sample ID")
    tumor=ann.loc[ann.sample_type_code.isin(["01","06","07"])].copy()
    genes=original.pseudogene.tolist()
    if not set(genes).issubset(expr.index) or not set(tumor.sample_id).issubset(expr.columns):
        raise ValueError("Original genes or tumor samples missing from expression matrix")
    baseline=calculate(expr,tumor,genes).set_index("pseudogene")
    old=original.set_index("pseudogene").loc[genes]
    fields={"mean_primary":"mean_primary","mean_metastatic":"mean_metastatic","median_primary":"median_primary","median_metastatic":"median_metastatic","pvalue":"pvalue","FDR":"FDR_BH","log2FC_metastatic_vs_primary":"mean_difference_metastatic_minus_primary"}
    for source,dest_col in fields.items():
        if not np.allclose(old[source],baseline.loc[genes,dest_col],rtol=1e-7,atol=1e-12):
            raise ValueError(f"Baseline mismatch in {source}. No new output written; verify input files.")
    multi=tumor.groupby("patient_id").size().loc[lambda x:x.gt(1)].index
    excluded=tumor.loc[tumor.patient_id.isin(multi)].copy()
    excluded["exclusion_reason"]="patient_has_multiple_tumor_samples"
    included=tumor.loc[~tumor.patient_id.isin(multi)].copy()
    if included.patient_id.duplicated().any(): raise ValueError("Repeated patients remain")
    result=calculate(expr,included,genes)
    compare=old[["mean_primary","mean_metastatic","log2FC_metastatic_vs_primary","pvalue","FDR"]].rename(columns={c:"original_"+c for c in ["mean_primary","mean_metastatic","log2FC_metastatic_vs_primary","pvalue","FDR"]}).join(result.set_index("pseudogene"),how="inner")
    compare["original_passes_FDR_0_05"]=compare.original_FDR.lt(.05)
    compare["original_passes_combined"]=compare.original_passes_FDR_0_05 & compare.original_log2FC_metastatic_vs_primary.abs().gt(.5)
    compare["FDR_status_changed"]=compare.original_passes_FDR_0_05.ne(compare.passes_FDR_0_05)
    compare["combined_status_changed"]=compare.original_passes_combined.ne(compare.passes_FDR_and_abs_difference_0_5)
    compare["effect_sign_changed"]=np.sign(compare.original_log2FC_metastatic_vs_primary).ne(np.sign(compare.mean_difference_metastatic_minus_primary))
    dest=args.outdir if args.outdir else root/"results"/"primary_metastatic_patient_sensitivity"
    dest.mkdir(parents=True,exist_ok=True)
    result.sort_values("FDR_BH").to_csv(dest/"patient_independent_comparison.csv",index=False,encoding="utf-8-sig")
    compare.to_csv(dest/"original_vs_patient_independent.csv",index_label="pseudogene",encoding="utf-8-sig")
    excluded.to_csv(dest/"excluded_samples.csv",index=False,encoding="utf-8-sig")
    included.to_csv(dest/"included_samples.csv",index=False,encoding="utf-8-sig")
    expr.loc[genes,included.sample_id].to_csv(dest/"patient_independent_expression.tsv.gz",sep="\t",compression="gzip",index_label="pseudogene")
    notes="""Sensitivity analysis, not replacement of original results.
All patients with more than one tumor sample (codes 01/06/07) are excluded, without choosing samples by expression or outcome.
Original gene selection is held fixed: no refiltering or new log transformation.
Expression scale: log2(normalized RSEM count + 1), per source metadata.
Two-sided Mann-Whitney U (SciPy method=auto); BH across all retained original genes.
The test compares distributions; it is not generally a test of means or medians alone.
Effect column is a difference of mean log-transformed expression values, not a ratio of arithmetic mean counts.
This addresses repeated observations only; clinical covariates and other confounding are not adjusted.
Old comparison and correlation-analysis files are not modified.
"""
    (dest/"README.txt").write_text(notes,encoding="utf-8")
    settings={"expression":str(expression_path.resolve()),"annotation":str(annotation_path.resolve()),"original_results":str(original_path.resolve()),"scipy_version":scipy.__version__,"pandas_version":pd.__version__,"n_genes":len(genes),"excluded_patients":sorted(multi.tolist())}
    (dest/"analysis_settings.json").write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding="utf-8")
    print("Baseline validation passed: original summary statistics, p-values and FDR reproduced.")
    print("Tumor samples before exclusion:",len(tumor))
    print("Excluded patients:",len(multi))
    print("Excluded tumor samples:",len(excluded))
    print("Primary samples retained:",int(included.sample_type_code.eq("01").sum()))
    print("Metastatic samples retained:",int(included.sample_type_code.isin(["06","07"]).sum()))
    print("Genes tested (fixed original selection):",len(genes))
    print("FDR < 0.05:",int(result.passes_FDR_0_05.sum()))
    print("FDR < 0.05 AND abs(mean difference) > 0.5:",int(result.passes_FDR_and_abs_difference_0_5.sum()))
    print("FDR status changes:",int(compare.FDR_status_changed.sum()))
    print("Combined status changes:",int(compare.combined_status_changed.sum()))
    print("Output:",dest.resolve())
    print("Original results unchanged.")

if __name__=="__main__":
    main()
