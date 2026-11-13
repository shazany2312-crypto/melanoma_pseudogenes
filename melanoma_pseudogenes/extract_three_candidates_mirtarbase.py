from pathlib import Path
import pandas as pd

def main():
    root=Path(__file__).resolve().parent
    name="miRTarBase_search_mirna_hsa-miR-186-5p.csv"
    paths=[p for p in root.rglob(name) if ".venv" not in p.parts]
    if len(paths)!=1:
        raise ValueError(f"Expected one {name}; found {len(paths)}: {paths}")
    data=pd.read_csv(paths[0])
    clean=lambda c:data[c].fillna("").astype(str).str.strip()
    mask=clean("miRNA").eq("hsa-miR-186-5p") & clean("Species (miRNA)").eq("Homo sapiens") & clean("Species (Target)").eq("Homo sapiens")
    human=data.loc[mask].copy()
    human["Target Gene"]=human["Target Gene"].str.strip()
    genes=["ZBTB7A","ZNF385A","MDGA1"]
    selected=human.loc[human["Target Gene"].isin(genes)].copy()
    dest=root/"results"/"mir186_target_analysis"/"candidate_annotation"
    dest.mkdir(parents=True,exist_ok=True)
    selected.to_csv(dest/"three_candidates_miRTarBase_records.csv",index=False)
    counts=pd.DataFrame({"target_gene":genes,"n_export_records":[int(human["Target Gene"].eq(g).sum()) for g in genes]})
    counts.to_csv(dest/"three_candidates_miRTarBase_coverage.csv",index=False)
    print("Export rows:",len(data))
    print("Human hsa-miR-186-5p rows:",len(human))
    print("Candidate records:",len(selected))
    cols=["MTI ID","Target Gene","Reporter Assay","Western Blot","qPCR","NGS","CLIP-Seq","Papers","Score"]
    print(selected[cols].to_string(index=False))
    print("Output:",dest.resolve())
    print("Method flags are database annotations, not proof of regulation in melanoma.")

if __name__=="__main__":
    main()
