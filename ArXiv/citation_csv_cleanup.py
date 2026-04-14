import pandas as pd

paper_df = pd.read_csv("arxiv_to_s2_mapping.csv")
new_papers_df = pd.read_csv("new_papers_from_references.csv")
citation_df = pd.read_csv("citation.csv")

# Ensure seed papers have internal IDs
if "paper_id" not in paper_df.columns:
    paper_df = paper_df.reset_index(drop=True)
    paper_df["paper_id"] = paper_df.index + 1

# Give new referenced papers fresh internal IDs
max_seed_id = paper_df["paper_id"].max()
new_papers_df = new_papers_df.drop_duplicates(subset=["s2_paperId"]).reset_index(drop=True)
new_papers_df["paper_id"] = range(max_seed_id + 1, max_seed_id + 1 + len(new_papers_df))

# Unified Paper table
paper_seed_cols = [
    "paper_id", "s2_paperId", "arxiv_title", "s2_year", "s2_venue",
    "arxiv_id_no_version", "s2_externalIds_json", "s2_publicationDate"
]
seed_paper_table = paper_df.copy()
seed_paper_table["title"] = seed_paper_table["s2_title"].fillna(seed_paper_table["arxiv_title"])
seed_paper_table["year"] = seed_paper_table["s2_year"]
seed_paper_table["venue"] = seed_paper_table["s2_venue"]
seed_paper_table["arxiv_id"] = seed_paper_table["arxiv_id_no_version"]
seed_paper_table["externalIds_json"] = seed_paper_table["s2_externalIds_json"]
seed_paper_table["publicationDate"] = seed_paper_table["s2_publicationDate"]
seed_paper_table = seed_paper_table[["paper_id", "s2_paperId", "title", "year", "venue", "arxiv_id", "externalIds_json", "publicationDate"]]

new_paper_table = new_papers_df.rename(columns={
    "title": "title",
    "year": "year",
    "venue": "venue",
    "arxiv_id": "arxiv_id",
    "externalIds_json": "externalIds_json",
    "publicationDate": "publicationDate",
})[["paper_id", "s2_paperId", "title", "year", "venue", "arxiv_id", "externalIds_json", "publicationDate"]]

paper_table = pd.concat([seed_paper_table, new_paper_table], ignore_index=True)
paper_table = paper_table.drop_duplicates(subset=["s2_paperId"], keep="first")

# Map S2 IDs to internal IDs
s2_to_internal = dict(zip(paper_table["s2_paperId"].astype(str), paper_table["paper_id"]))

citation_df["target_paper_id"] = citation_df["target_s2_paperId"].astype(str).map(s2_to_internal)
citation_df["source_paper_id"] = citation_df["source_paper_id"]

citation_fk = citation_df[["source_paper_id", "target_paper_id"]].dropna().drop_duplicates()

# Exclude seed papers that have 0 references in S2
seed_ids_with_refs = set(citation_fk["source_paper_id"].unique())
paper_table = paper_table[
    (~paper_table["paper_id"].isin(seed_paper_table["paper_id"])) |
    (paper_table["paper_id"].isin(seed_ids_with_refs))
].copy()

paper_table.to_csv("paper_table_full.csv", index=False)
new_papers_df.to_csv("new_papers_with_internal_ids.csv", index=False)
citation_fk.to_csv("citation_fk.csv", index=False)

print("Wrote:")
print("- paper_table_full.csv")
print("- new_papers_with_internal_ids.csv")
print("- citation_fk.csv")