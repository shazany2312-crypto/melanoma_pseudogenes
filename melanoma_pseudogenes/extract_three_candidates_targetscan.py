from pathlib import Path
import pandas as pd

def main():
    root=Path(__file__).resolve().parent
    name="TargetScan8.0__miR-186-5p.Human.predicted_targets.txt"
    paths=[p for p in root.rglob(name) if ".venv" not in p.parts]
    if len(paths)!=1:
        raise ValueError(f"Expected one {name}, found {len(paths)}: {paths}")
    data=pd.read_csv(paths[0],sep="\t")
    genes=["ZBTB7A","ZNF385A","MDGA1"]
    result=data.loc[data["Target gene"].isin(genes)].copy()
    if set(result["Target gene"])!=set(genes):
        raise ValueError("Some candidate genes are missing")
    dest=root/"results"/"mir186_target_analysis"/"candidate_annotation"
    dest.mkdir(parents=True,exist_ok=True)
    result.to_csv(dest/"three_candidates_TargetScan_records.csv",index=False)
    columns=["Target gene","Representative transcript","Conserved sites total","Conserved 8mer sites","Conserved 7mer-m8 sites","Conserved 7mer-A1 sites","Cumulative weighted context++ score"]
    print(result[columns].to_string(index=False))
    print("Saved records:",len(result))
    print("Output:",dest.resolve())
    print("These are computational predictions, not experimental validations.")

if __name__=="__main__":
    main()
