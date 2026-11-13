from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata


def project_root():
    here=Path(__file__).resolve().parent
    for root in [here,here.parent]:
        if (root/"results"/"primary_metastatic_patient_sensitivity"/"included_samples.csv").is_file():
            return root
    raise FileNotFoundError("Save script in project root or scripts folder. Patient sensitivity outputs must exist.")


def main():
    parser=argparse.ArgumentParser(description="Descriptive gene-profile clustering: Spearman distance and average linkage")
    parser.add_argument("--root",type=Path)
    parser.add_argument("--expression",type=Path)
    parser.add_argument("--annotation",type=Path)
    parser.add_argument("--outdir",type=Path)
    args=parser.parse_args()
    root=args.root.resolve() if args.root else project_root()
    source=root/"results"/"primary_metastatic_patient_sensitivity"
    ep=args.expression if args.expression else source/"patient_independent_expression.tsv.gz"
    ap=args.annotation if args.annotation else source/"included_samples.csv"
    for path in [ep,ap]:
        if not path.is_file():raise FileNotFoundError(str(path))
    expr=pd.read_csv(ep,sep="\t",index_col=0)
    ann=pd.read_csv(ap,dtype=str)
    if expr.index.has_duplicates or expr.columns.has_duplicates or ann.sample_id.duplicated().any():
        raise ValueError("Duplicate gene/sample IDs")
    if ann.patient_id.duplicated().any():raise ValueError("Repeated patients remain")
    if set(expr.columns)!=set(ann.sample_id):raise ValueError("Expression and annotation sample IDs differ")
    ann["sample_type_code"]=ann.sample_type_code.str.zfill(2)
    if not ann.sample_type_code.isin(["01","06","07"]).all():raise ValueError("Unexpected sample type")
    ann["display_group"]=np.where(ann.sample_type_code.eq("01"),"Primary","Metastatic")
    ann["group_order"]=ann.display_group.map({"Primary":0,"Metastatic":1})
    ann=ann.sort_values(["group_order","sample_id"]).set_index("sample_id")
    expr=expr.loc[:,ann.index].astype(float)
    if not np.isfinite(expr.to_numpy()).all():raise ValueError("Nonfinite expression values; no samples silently removed")
    sd=expr.std(axis=1,ddof=0)
    constant=expr.index[sd.eq(0)]
    expr=expr.loc[sd.gt(0)]
    if len(expr)<2:raise ValueError("Fewer than two variable genes")
    ranks=np.vstack([rankdata(row,method="average") for row in expr.to_numpy()])
    rho=np.corrcoef(ranks)
    rho=np.clip((rho+rho.T)/2,-1,1)
    np.fill_diagonal(rho,1.0)
    dist=1-rho
    np.fill_diagonal(dist,0.0)
    tree=linkage(squareform(dist,checks=True),method="average",optimal_ordering=True)
    order=leaves_list(tree)
    genes=expr.index[order]
    z=expr.sub(expr.mean(axis=1),axis=0).div(expr.std(axis=1,ddof=0),axis=0)
    colors={"Primary":"#4C72B0","Metastatic":"#DD8452"}
    bar=ann.display_group.map(colors).rename("Sample type")
    dest=args.outdir if args.outdir else root/"results"/"pseudogene_profile_clustering"
    dest.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style="white",font_scale=.8)
    grid=sns.clustermap(z,row_linkage=tree,col_cluster=False,col_colors=bar,cmap="RdBu_r",vmin=-2.5,vmax=2.5,xticklabels=False,yticklabels=True,figsize=(15,12),dendrogram_ratio=(.15,.06),cbar_kws={"label":"Within-gene z-score (display capped at +/-2.5)"})
    grid.ax_heatmap.set_xlabel("Samples: Primary first, then Metastatic; ID order within each group (not clustered)")
    grid.ax_heatmap.set_ylabel("Processed pseudogenes")
    grid.ax_heatmap.tick_params(axis="y",labelsize=8)
    n_primary=int(ann.display_group.eq("Primary").sum())
    grid.ax_heatmap.axvline(n_primary,color="black",lw=.8)
    grid.fig.suptitle(f"Gene expression profiles: {len(expr)} pseudogenes, {expr.shape[1]} samples\nGenes: Spearman distance (1-rho), average linkage; no fixed cluster count",y=.995,fontsize=12)
    grid.ax_col_dendrogram.legend(handles=[Patch(facecolor=c,label=g) for g,c in colors.items()],loc="center",ncol=2,frameon=False)
    grid.fig.savefig(dest/"pseudogene_expression_heatmap.jpg",dpi=180,facecolor="white",pil_kwargs={"quality":95})
    grid.fig.savefig(dest/"pseudogene_expression_heatmap.pdf",facecolor="white")
    plt.close(grid.fig)
    fig,ax=plt.subplots(figsize=(15,6))
    dendrogram(tree,labels=expr.index.tolist(),leaf_rotation=90,leaf_font_size=8,color_threshold=0,above_threshold_color="#333333",ax=ax)
    ax.set_ylabel("Average-linkage distance (1 - Spearman rho)")
    ax.set_title("Hierarchical clustering of pseudogene expression profiles; no cluster cut")
    fig.tight_layout()
    fig.savefig(dest/"pseudogene_dendrogram.jpg",dpi=180,facecolor="white",pil_kwargs={"quality":95})
    fig.savefig(dest/"pseudogene_dendrogram.pdf",facecolor="white")
    plt.close(fig)
    pd.DataFrame(rho,index=expr.index,columns=expr.index).loc[genes,genes].to_csv(dest/"gene_spearman_correlations.csv",index_label="pseudogene")
    z.loc[genes].to_csv(dest/"heatmap_zscores.csv",index_label="pseudogene")
    expr.loc[genes].to_csv(dest/"heatmap_input_expression.csv",index_label="pseudogene")
    pd.DataFrame({"leaf_position_1based":range(1,len(genes)+1),"pseudogene":genes}).to_csv(dest/"gene_leaf_order.csv",index=False)
    pd.DataFrame(tree,columns=["left_node","right_node","distance","n_leaves"]).to_csv(dest/"gene_linkage.csv",index_label="merge_index")
    ann.drop(columns="group_order").to_csv(dest/"heatmap_sample_order.csv",index_label="sample_id")
    pd.DataFrame({"pseudogene":constant,"reason":"constant_expression"}).to_csv(dest/"excluded_constant_genes.csv",index=False)
    settings={"expression_path":str(ep.resolve()),"annotation_path":str(ap.resolve()),"distance":"1 - Spearman rho; average ranks for ties","linkage":"average; optimal leaf ordering","zscore_ddof":0,"display_color_limits":[-2.5,2.5],"sample_clustering":False,"cluster_cut":None,"n_genes":len(expr),"n_samples":expr.shape[1]}
    (dest/"clustering_settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    notes="""Descriptive clustering of pseudogenes, not patients.
Input: fixed previously selected gene set, patients with multiple tumor samples excluded.
No selection by differential-expression p-values, ANXA family membership, or group separation.
No additional log transformation. Within-gene z-scores are for display only.
Spearman correlations use original input expression with average ranks for ties.
Genes are clustered by 1-rho and average linkage. Negative correlations imply greater distance, not similarity.
Samples are deliberately NOT clustered: group blocks and sample-ID order are imposed for display.
Heatmap colors are capped visually at +/-2.5; saved values are not clipped.
No fixed number of clusters or statistical significance is claimed.
Pooled co-expression can reflect between-group effects or tissue composition; it is not proof of shared regulation or ceRNA interaction.
"""
    (dest/"README.txt").write_text(notes,encoding="utf-8")
    print("Input genes:",len(sd))
    print("Excluded constant genes:",len(constant))
    print("Clustered pseudogenes:",len(expr))
    print("Samples:",expr.shape[1])
    print("Primary:",n_primary)
    print("Metastatic:",int(ann.display_group.eq("Metastatic").sum()))
    print("Gene clustering: 1 - Spearman rho; average linkage.")
    print("Sample clustering: NOT performed; fixed group/ID order.")
    print("Fixed cluster count: NOT assigned.")
    print("Output:",dest.resolve())

if __name__=="__main__":
    main()
