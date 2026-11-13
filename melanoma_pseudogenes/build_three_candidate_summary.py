from pathlib import Path
import argparse
import pandas as pd

GENES=["ZBTB7A","ZNF385A","MDGA1"]
def locate(root,name):
    matches=[p for p in root.rglob(name) if ".venv" not in p.parts and ".git" not in p.parts]
    if len(matches)!=1:
        raise ValueError(f"Expected one {name}; found {len(matches)}: {matches}")
    return matches[0]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--loo",type=Path)
    parser.add_argument("--outdir",type=Path)
    args=parser.parse_args()
    root=args.root.resolve()
    sources={n:locate(root,n) for n in ["paired_target_results.csv","TargetScan8.0__miR-186-5p.Human.predicted_targets.txt","miRTarBase_search_mirna_hsa-miR-186-5p.csv"]}
    sources["loo_summary.csv"]=args.loo if args.loo else locate(root,"loo_summary.csv")
    r=pd.read_csv(sources["paired_target_results.csv"])
    t=pd.read_csv(sources["TargetScan8.0__miR-186-5p.Human.predicted_targets.txt"],sep="\t")
    m=pd.read_csv(sources["miRTarBase_search_mirna_hsa-miR-186-5p.csv"])
    l=pd.read_csv(sources["loo_summary.csv"])
    r=r.loc[r.target_gene.isin(GENES)].copy()
    t=t.loc[t["Target gene"].isin(GENES)].copy()
    for col in ["miRNA","Species (miRNA)","Species (Target)","Target Gene"]:
        m[col]=m[col].fillna("").astype(str).str.strip()
    m=m.loc[m.miRNA.eq("hsa-miR-186-5p") & m["Species (miRNA)"].eq("Homo sapiens") & m["Species (Target)"].eq("Homo sapiens") & m["Target Gene"].isin(GENES)]
    if r.duplicated(["target_gene","group"]).any() or l.duplicated(["target_gene","group","relation"]).any():
        raise ValueError("Duplicate correlation or sensitivity rows")
    short=[];long=[]
    methods=["Reporter Assay","Western Blot","qPCR","Microarray","NGS","CLASH","pSILAC","CLIP-Seq","Other"]
    for gene in GENES:
        ts=t.loc[t["Target gene"].eq(gene)]
        if len(ts)!=1:
            raise ValueError(f"Expected one TargetScan record for {gene}")
        ts=ts.iloc[0]
        mr=m.loc[m["Target Gene"].eq(gene)]
        if mr.empty:
            raise ValueError(f"No human miRTarBase record for {gene}")
        item={"target_gene":gene,"targetscan_transcript":ts["Representative transcript"],"targetscan_conserved_sites":ts["Conserved sites total"],"targetscan_conserved_8mer":ts["Conserved 8mer sites"],"targetscan_conserved_7mer_m8":ts["Conserved 7mer-m8 sites"],"targetscan_conserved_7mer_A1":ts["Conserved 7mer-A1 sites"],"targetscan_cumulative_context_score":ts["Cumulative weighted context++ score"],"mirtarbase_ids":"; ".join(mr["MTI ID"].astype(str)),"mirtarbase_categories":"; ".join(mr.Score.astype(str).unique()),"mirtarbase_methods":"; ".join(c for c in methods if pd.to_numeric(mr[c],errors="raise").gt(0).any()),"mirtarbase_papers_field":"; ".join(mr.Papers.astype(str)),"source_review_status":"publication identified; exact pair-level evidence not fully traced" if gene=="ZNF385A" else "PENDING: identify CLIP dataset, cell type and original source","source_pmid":"20371350" if gene=="ZNF385A" else "","source_cell_context":"HEK293 (source study; not melanoma)" if gene=="ZNF385A" else "not yet verified","interpretation":"Exploratory candidate; ceRNA mechanism not demonstrated"}
        for group in ["all_main","primary","metastatic"]:
            rr=r.loc[r.target_gene.eq(gene)&r.group.eq(group)]
            if len(rr)!=1: raise ValueError(f"Missing {gene}/{group}")
            rr=rr.iloc[0]
            item[f"{group}_candidate_pattern"]=rr.candidate_pattern
            for rel in ["mirna","pseudogene"]:
                item[f"{group}_{rel}_rho"]=rr[f"spearman_rho_{rel}"]
                item[f"{group}_{rel}_BH_adjusted_p"]=rr[f"FDR_BH_all_current_tests_{rel}"]
                if group=="all_main": continue
                ll=l.loc[l.target_gene.eq(gene)&l.group.eq(group)&l.relation.eq(rel)]
                if len(ll)!=1: raise ValueError(f"Missing sensitivity {gene}/{group}/{rel}")
                ll=ll.iloc[0]
                if abs(float(ll.rho_full)-float(rr[f"spearman_rho_{rel}"]))>1e-10 or int(ll.n_full)!=int(rr[f"n_samples_{rel}"]):
                    raise ValueError(f"Sensitivity baseline mismatch: {gene}/{group}/{rel}")
                record={"target_gene":gene,"group":group,"relation":rel,"n":int(ll.n_full),"rho":ll.rho_full,"BH_adjusted_p":rr[f"FDR_BH_all_current_tests_{rel}"],"rho_loo_min":ll.rho_loo_min,"rho_loo_max":ll.rho_loo_max,"max_abs_delta_rho":ll.max_abs_delta_rho,"n_sign_changes":int(ll.n_sign_changes),"most_influential_sample_id":ll.most_influential_sample_id}
                long.append(record)
        gl=l.loc[l.target_gene.eq(gene)&l.group.isin(["primary","metastatic"])]
        item["max_single_sample_abs_delta_rho"]=gl.max_abs_delta_rho.max()
        item["total_sign_changes"]=int(gl.n_sign_changes.sum())
        short.append(item)
    dest=args.outdir if args.outdir else sources["paired_target_results.csv"].parent/"candidate_summary"
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(short).to_csv(dest/"candidate_overview_3_genes.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(long).to_csv(dest/"candidate_correlations_12_tests.csv",index=False,encoding="utf-8-sig")
    notes="""# Candidate summary: scope and unfinished work

- This script merges existing results; it does not recalculate correlations, p-values or FDR.
- BH-adjusted p-values refer to the original family of 5292 tests, not 12 selected tests.
- Prior miRNA selection is not accounted for in the current correction.
- Leave-one-out checks concern coefficient sensitivity to one observation, not significance retention or causal validity.
- TargetScan predictions concern representative transcripts; expression was quantified at gene level.
- miRTarBase method flags and categories are database annotations, not proof of regulation in melanoma.
- ZNF385A / MIRT021123: source publication identified as Hafner et al., Cell 2010, PMID 20371350; exact pair/site-level evidence not fully traced. Source study uses HEK293.
- PENDING: ZBTB7A / MIRT1643119 and MDGA1 / MIRT1640791: identify original CLIP datasets, cell types and experimental conditions. Website access problems interrupted review; do not mark completed.
- No ceRNA mechanism has been demonstrated by this analysis.
"""
    (dest/"README_pending_checks.md").write_text(notes,encoding="utf-8")
    print("Validation passed: 12 sensitivity baselines match saved correlations.")
    print("Candidate genes:",len(short))
    print("Detailed correlation rows:",len(long))
    print("Pending CLIP source checks: ZBTB7A, MDGA1")
    print("Output:",dest.resolve())

if __name__=="__main__":
    main()