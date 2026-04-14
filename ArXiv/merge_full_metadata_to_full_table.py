import pandas as pd

seed_df = pd.read_csv("arxiv_to_s2_mapping.csv")
ref_full_df = pd.read_csv("referenced_papers_full.csv")

# Keep only successful referenced-paper fetches
ref_full_df = ref_full_df[ref_full_df["found"] == 1].copy()

# Ensure seed papers have internal paper_id
if "paper_id" not in seed_df.columns:
    seed_df = seed_df.reset_index(drop=True)
    seed_df["paper_id"] = seed_df.index + 1

# Build normalized seed paper slice
seed_norm = pd.DataFrame({
    "paper_id": seed_df["paper_id"],
    "s2_paperId": seed_df["s2_paperId"],
    "title": seed_df["s2_title"].fillna(seed_df["arxiv_title"]),
    "abstract": None,
    "year": seed_df["s2_year"],
    "venue": seed_df["s2_venue"],
    "publicationDate": seed_df["s2_publicationDate"],
    "authors_json": None,
    "doi": None,
    "arxiv_id": seed_df["arxiv_id_no_version"],
    "externalIds_json": seed_df["s2_externalIds_json"],
    "referenceCount": seed_df["s2_referenceCount"],
    "citationCount": seed_df["s2_citationCount"],
    "influentialCitationCount": None,
    "isOpenAccess": None,
    "openAccessPdf_url": None,
    "fieldsOfStudy_json": seed_df["s2_fieldsOfStudy_json"],
    "s2FieldsOfStudy_json": seed_df["s2_s2FieldsOfStudy_json"],
    "s2_url": None,
})

# Build normalized referenced paper slice
ref_norm = ref_full_df[[
    "s2_paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "publicationDate",
    "authors_json",
    "doi",
    "arxiv_id",
    "externalIds_json",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf_url",
    "fieldsOfStudy_json",
    "s2FieldsOfStudy_json",
    "s2_url",
]].copy()

# Remove referenced papers already present among seed papers
seed_s2_ids = set(seed_norm["s2_paperId"].dropna().astype(str))
ref_norm = ref_norm[~ref_norm["s2_paperId"].astype(str).isin(seed_s2_ids)].copy()

# Assign new internal paper IDs
max_seed_id = int(seed_norm["paper_id"].max())
ref_norm = ref_norm.drop_duplicates(subset=["s2_paperId"]).reset_index(drop=True)
ref_norm["paper_id"] = range(max_seed_id + 1, max_seed_id + 1 + len(ref_norm))

# Align column order
cols = [
    "paper_id",
    "s2_paperId",
    "title",
    "abstract",
    "year",
    "venue",
    "publicationDate",
    "authors_json",
    "doi",
    "arxiv_id",
    "externalIds_json",
    "referenceCount",
    "citationCount",
    "influentialCitationCount",
    "isOpenAccess",
    "openAccessPdf_url",
    "fieldsOfStudy_json",
    "s2FieldsOfStudy_json",
    "s2_url",
]
seed_norm = seed_norm[cols]
ref_norm = ref_norm[cols]

paper_table_full = pd.concat([seed_norm, ref_norm], ignore_index=True)
paper_table_full = paper_table_full.drop_duplicates(subset=["s2_paperId"], keep="first")

paper_table_full.to_csv("paper_table_full_rich.csv", index=False)

print("Wrote paper_table_full_rich.csv")