import pandas as pd

paper_table = pd.read_csv("paper_table_full_rich.csv")
citation_df = pd.read_csv("citation.csv")

s2_to_internal = dict(zip(
    paper_table["s2_paperId"].astype(str),
    paper_table["paper_id"]
))

citation_df["target_paper_id"] = citation_df["target_s2_paperId"].astype(str).map(s2_to_internal)
citation_df["source_paper_id"] = citation_df["source_paper_id"]

citation_fk = citation_df[["source_paper_id", "target_paper_id"]].dropna().drop_duplicates()
citation_fk.to_csv("citation_fk_rich.csv", index=False)

print("Wrote citation_fk_rich.csv")