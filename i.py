
# %%
import glob
import polars as pl

tsv_files = glob.glob("dataset/**/*.tsv", recursive=True)

for file in tsv_files:
    try:
        df = pl.read_csv(file, separator="\t", ignore_errors=True)
        print(f"\n--- {file} ---")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns}")
    except Exception as e:
        print(f"Error reading {file}: {e}")