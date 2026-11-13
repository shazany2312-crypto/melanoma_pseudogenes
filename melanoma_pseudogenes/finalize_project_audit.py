from pathlib import Path
import argparse
import hashlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd

REQUIRED={
"original_comparison":"results/tables/primary_vs_metastatic_pseudogene_expression.tsv",
"patient_comparison":"results/primary_metastatic_patient_sensitivity/patient_independent_comparison.csv",
"patient_samples":"results/primary_metastatic_patient_sensitivity/included_samples.csv",
"excluded_samples":"results/primary_metastatic_patient_sensitivity/excluded_samples.csv",
"patient_expression":"results/primary_metastatic_patient_sensitivity/patient_independent_expression.tsv.gz",
"gene_correlations":"results/pseudogene_profile_clustering/gene_spearman_correlations.csv",
"within_group_correlations":"results/anxa2p_within_group_correlations/anxa2p_correlations_by_group.csv",
"target_correlations":"results/mir186_target_analysis/paired_target_results.csv",
"candidate_overview":"results/mir186_target_analysis/candidate_summary/candidate_overview_3_genes.csv",
"candidate_details":"results/mir186_target_analysis/candidate_summary/candidate_correlations_12_tests.csv",
"candidate_loo":"results/mir186_target_analysis/leave_one_out_sensitivity/loo_summary.csv",
"targetscan_extract":"results/mir186_target_analysis/candidate_annotation/three_candidates_TargetScan_records.csv",
"mirtarbase_extract":"results/mir186_target_analysis/candidate_annotation/three_candidates_miRTarBase_records.csv",
}
FIGURES=[
"results/pseudogene_profile_clustering/pseudogene_dendrogram.jpg",
"results/pseudogene_profile_clustering/pseudogene_dendrogram.pdf",
"results/pseudogene_profile_clustering/pseudogene_expression_heatmap.jpg",
"results/pseudogene_profile_clustering/pseudogene_expression_heatmap.pdf",
"results/mir186_target_analysis/heatmap_visualization/candidate_heatmap.jpg",
]+[f"results/mir186_target_analysis/candidate_scatterplots/{gene}_scatterplots.jpg" for gene in ["MDGA1","ZBTB7A","ZNF385A"]]


def find_root():
    here=Path(__file__).resolve().parent
    for root in [here,here.parent]:
        if (root/"results"/"tables"/"primary_vs_metastatic_pseudogene_expression.tsv").is_file(): return root
    raise FileNotFoundError("Save in project root or scripts folder, or pass --root")


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description="Read-only audit of analysis inputs/outputs; writes audit documents only")
    parser.add_argument("--root",type=Path)
    args=parser.parse_args()
    root=args.root.resolve() if args.root else find_root()
    dest=root/"results"/"final_project_audit"
    dest.mkdir(parents=True,exist_ok=True)
    checks=[];manifest=[];tables={}
    def record(name,ok,detail):
        checks.append(dict(check=name,status="PASS" if ok else "NEEDS_REVIEW",details=str(detail)))
    def test(name,func):
        try:
            value=func()
            record(name,bool(value),"Condition satisfied" if value else "Condition not satisfied")
        except Exception as error:
            record(name,False,f"{type(error).__name__}: {error}")
    for role,relative in REQUIRED.items():
        path=root/relative
        exists=path.is_file() and path.stat().st_size>0
        record("file_"+role,exists,relative)
        if not exists: continue
        manifest.append(dict(role=role,relative_path=relative,size_bytes=path.stat().st_size,sha256=digest(path)))
        try:
            frame=pd.read_csv(path,sep="\t" if ".tsv" in path.name else ",",index_col=0 if role in ["patient_expression","gene_correlations"] else None)
            tables[role]=frame
            manifest[-1].update(rows=len(frame),columns=len(frame.columns))
        except Exception as error: record("read_"+role,False,str(error))
    for relative in FIGURES:
        path=root/relative
        ok=path.is_file() and path.stat().st_size>0
        record("figure_"+path.name+"_"+path.parent.name,ok,relative)
        if ok:manifest.append(dict(role="figure",relative_path=relative,size_bytes=path.stat().st_size,sha256=digest(path)))
    test("original_42_unique_genes",lambda:len(tables["original_comparison"])==42 and tables["original_comparison"].pseudogene.nunique()==42)
    test("patient_42_same_genes",lambda:set(tables["patient_comparison"].pseudogene)==set(tables["original_comparison"].pseudogene) and len(tables["patient_comparison"])==42)
    test("patient_samples_467_no_repeats",lambda:len(tables["patient_samples"])==467 and not tables["patient_samples"].patient_id.duplicated().any() and not tables["patient_samples"].sample_id.duplicated().any())
    test("patient_groups_102_365",lambda:pd.to_numeric(tables["patient_samples"].sample_type_code).eq(1).sum()==102 and pd.to_numeric(tables["patient_samples"].sample_type_code).isin([6,7]).sum()==365)
    test("excluded_6_samples_3_patients",lambda:len(tables["excluded_samples"])==6 and tables["excluded_samples"].patient_id.nunique()==3 and not set(tables["excluded_samples"].patient_id)&set(tables["patient_samples"].patient_id))
    test("patient_expression_42x467_finite",lambda:tables["patient_expression"].shape==(42,467) and np.isfinite(tables["patient_expression"].to_numpy(float)).all() and set(tables["patient_expression"].columns)==set(tables["patient_samples"].sample_id) and set(tables["patient_expression"].index)==set(tables["patient_comparison"].pseudogene))
    test("patient_effect_equals_mean_difference",lambda:np.allclose(tables["patient_comparison"].mean_difference_metastatic_minus_primary,tables["patient_comparison"].mean_metastatic-tables["patient_comparison"].mean_primary))
    test("patient_combined_candidates",lambda:set(tables["patient_comparison"].loc[tables["patient_comparison"].FDR_BH.lt(.05)&tables["patient_comparison"].mean_difference_metastatic_minus_primary.abs().gt(.5),"pseudogene"])=={"ANXA2P1","ANXA2P2","ANXA2P3"})
    def check_matrix():
        c=tables["gene_correlations"]
        return c.shape==(42,42) and set(c.index)==set(tables["patient_expression"].index) and list(c.index)==list(c.columns) and np.isfinite(c.to_numpy(float)).all() and np.allclose(c,c.T) and np.allclose(np.diag(c),1) and (np.abs(c.to_numpy())<=1+1e-12).all()
    test("gene_correlation_matrix",check_matrix)
    def check_within():
        w=tables["within_group_correlations"]
        selected=w[w.group.ne("all")]
        return len(selected)==6 and not selected.duplicated(["group","gene_1","gene_2"]).any() and set(w.group)=={"all","primary","metastatic"} and set(w.loc[w.group.eq("primary"),"n"])=={102} and set(w.loc[w.group.eq("metastatic"),"n"])=={365} and selected.BH_adjusted_p_six_within_group_tests.between(0,1).all()
    test("six_within_group_tests",check_within)
    def check_pooled():
        c=tables["gene_correlations"];w=tables["within_group_correlations"]
        return all(np.isclose(row.spearman_rho,c.loc[row.gene_1,row.gene_2],atol=1e-10,rtol=0) for row in w[w.group.eq("all")].itertuples())
    test("pooled_ANXA_correlations_match_clustering",check_pooled)
    def check_target():
        r=tables["target_correlations"]
        return len(r)==2646 and r.target_gene.nunique()==882 and not r.duplicated(["target_gene","group"]).any() and all(set(r.loc[r.group.eq(g),"n_samples_mirna"])=={n} and set(r.loc[r.group.eq(g),"n_samples_pseudogene"])=={n} for g,n in [("all_main",446),("primary",96),("metastatic",350)])
    test("target_analysis_separate_446_sample_cohort",check_target)
    expected={"ZBTB7A","ZNF385A","MDGA1"}
    test("three_candidate_overview",lambda:len(tables["candidate_overview"])==3 and set(tables["candidate_overview"].target_gene)==expected)
    test("12_candidate_detail_rows",lambda:len(tables["candidate_details"])==12 and not tables["candidate_details"].duplicated(["target_gene","group","relation"]).any())
    def check_loo():
        details=tables["candidate_details"].set_index(["target_gene","group","relation"])
        loo=tables["candidate_loo"].set_index(["target_gene","group","relation"])
        loo=loo.loc[details.index]
        return len(loo)==12 and np.allclose(details.rho,loo.rho_full) and np.allclose(details.max_abs_delta_rho,loo.max_abs_delta_rho)
    test("candidate_summary_matches_LOO",check_loo)
    test("targetscan_three_genes",lambda:set(tables["targetscan_extract"]["Target gene"])==expected)
    test("mirtarbase_three_human_pairs",lambda:set(tables["mirtarbase_extract"]["Target Gene"])==expected and tables["mirtarbase_extract"].miRNA.eq("hsa-miR-186-5p").all() and tables["mirtarbase_extract"]["Species (Target)"].eq("Homo sapiens").all())
    scripts=[]
    skip={".venv",".git","__pycache__","final_project_audit"}
    for path in sorted(root.rglob("*.py")):
        if any(part in skip for part in path.relative_to(root).parts): continue
        scripts.append(dict(filename=path.name,relative_path=str(path.relative_to(root)),size_bytes=path.stat().st_size,sha256=digest(path)))
    script_table=pd.DataFrame(scripts,columns=["filename","relative_path","size_bytes","sha256"])
    script_table["duplicate_basename"]=script_table.filename.duplicated(keep=False)
    script_table.to_csv(dest/"script_inventory.csv",index=False,encoding="utf-8-sig")
    duplicated=script_table.loc[script_table.duplicate_basename,"relative_path"].tolist()
    record("duplicate_script_names",not duplicated,"None" if not duplicated else "; ".join(duplicated))
    pending=[
      dict(task="ZBTB7A CLIP source",status="PENDING",record="MIRT1643119",next_step="Identify original AGO-CLIP dataset, cell type and conditions; website access postponed"),
      dict(task="MDGA1 CLIP source",status="PENDING",record="MIRT1640791",next_step="Identify original AGO-CLIP dataset, cell type and conditions; website access postponed"),
      dict(task="ZNF385A pair/site-level evidence",status="PARTIAL",record="MIRT021123; PMID 20371350",next_step="Publication identified; exact pair/site evidence not fully traced"),
    ]
    pd.DataFrame(pending).to_csv(dest/"pending_evidence_checks.csv",index=False,encoding="utf-8-sig")
    check_table=pd.DataFrame(checks)
    check_table.to_csv(dest/"technical_checks.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(manifest).to_csv(dest/"final_file_manifest.csv",index=False,encoding="utf-8-sig")
    env={"python":platform.python_version(),"platform":platform.platform()}
    for name in ["numpy","pandas","scipy","statsmodels","matplotlib","seaborn"]:
        try:env[name]=version(name)
        except PackageNotFoundError:env[name]="not installed"
    (dest/"environment.json").write_text(json.dumps(env,ensure_ascii=False,indent=2),encoding="utf-8")
    guide="""# Порядок воспроизведения и итоговые результаты

## Область проверки
Этот отчёт проверяет наличие и часть согласованности сохранённых результатов. Он не запускает всю цепочку с нуля и не доказывает корректность всех методов. PASS означает выполнение конкретных технических условий, а не подтверждение ceRNA-механизма.
Проверка записывает только файлы в results/final_project_audit. Исходные данные, скрипты и результаты анализа не изменяются.

## Два разных набора образцов
1. Общее сравнение: первоначально 104 первичные опухоли и 369 метастатических образцов. Уточняющая версия: 102 и 365, без пациентов с повторными опухолевыми образцами. Для кластеризации использована уточняющая версия, 42 ранее отобранных псевдогена.
2. Анализ miR-186-5p: отдельный сопоставленный набор 446 образцов, 96 первичных и 350 метастатических. Его нельзя заменять общей матрицей из 467 образцов.

## Шкала
TCGA.SKCM.sampleMap/HiSeqV2: log2(normalized RSEM count + 1), согласно метаданным UCSC Xena. Повторное логарифмирование не требуется.
Разность средних значений этой шкалы не является логарифмом отношения арифметических средних исходных counts.

## Порядок завершающих скриптов
Предусловия: исходная подготовка данных, аннотация и ранние результаты уже существуют. Ранние скрипты не запускайте вслепую: часть из них перезаписывает результаты.
1. compare_primary_metastatic_patient_sensitivity.py
2. cluster_pseudogene_profiles.py
3. check_anxa2p_correlations_by_group.py
Для блока miR-186 после подготовки paired_target_results.csv:
1. plot_mir186_candidate_heatmap.py
2. plot_mir186_candidate_scatterplots.py
3. check_mir186_leave_one_out.py
4. extract_three_candidates_targetscan.py
5. extract_three_candidates_mirtarbase.py
6. build_three_candidate_summary.py
В конце: finalize_project_audit.py.
Полный список реальных расположений скриптов находится в script_inventory.csv. Дубликаты имён требуют ручной сверки, а не автоматического удаления.

## Основные итоговые папки
- results/primary_metastatic_patient_sensitivity: сравнение без повторных пациентов и сопоставление со старой версией.
- results/pseudogene_profile_clustering: экспрессионная тепловая карта, дендрограмма, матрица корреляций. Группируются гены, не пациенты; число кластеров не фиксировано.
- results/anxa2p_within_group_correlations: три пары в двух группах; BH для шести тестов, разведочный анализ после выбора семейства.
- results/mir186_target_analysis/candidate_summary: сводка трёх кандидатов и 12 связей.

## Ограничения
BH для 5292 исходных корреляций не учитывает предшествующий поиск микроРНК и не гарантирует FDR для итогового списка тройных механизмов.
Leave-one-out оценивает влияние одного образца на коэффициент, не сохранение FDR.
TargetScan относится к представительным транскриптам, экспрессия — к генному уровню.
Высокопроизводительная поддержка miRTarBase не равна индивидуальной функциональной валидации в меланоме.
Общие и внутригрупповые корреляции не исключают техническую неоднозначность картирования сходных последовательностей, влияние клеточного состава и другие смешивающие факторы.
Незавершённые проверки перечислены в pending_evidence_checks.csv.

## Как читать завершение
Если technical_checks.csv содержит NEEDS_REVIEW, проверьте эти строки. Даже при всех PASS остаются ограничения и открытые проверки источников. Проверка среды и файлов не заменяет полный чистый перезапуск проекта.
"""
    (dest/"PROJECT_RUN_GUIDE.md").write_text(guide,encoding="utf-8")
    n_pass=int(check_table.status.eq("PASS").sum());n_review=int(check_table.status.eq("NEEDS_REVIEW").sum())
    print("Technical checks passed:",n_pass)
    print("Technical checks needing review:",n_review)
    print("Pending evidence items:",len(pending))
    print("Analysis files modified: 0")
    print("Output:",dest.resolve())
    if n_review:
        print("Review these checks:")
        print(check_table.loc[check_table.status.eq("NEEDS_REVIEW"),["check","details"]].to_string(index=False))
        print("AUDIT STATUS: NEEDS REVIEW (audit ran successfully; project not certified complete)")
    else:print("AUDIT STATUS: TECHNICAL CHECKS PASSED (not a full clean rerun or biological validation)")

if __name__=="__main__":
    main()
