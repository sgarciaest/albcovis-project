import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import traceback

# Import your function and model
from albcovis.services.cover_data_aggregator import CoverDataAggregator
from albcovis.models.cover_data import CoverDataModel

agg = CoverDataAggregator()

# File paths
data_pool_file = Path("./data/source/data_pool_20250813.csv")
dev_set_file = Path("./data/source/dev_set.json")
analysis_set_file = Path("./data/source/analysis_set.json")

# Load data pool (CSV with columns: mbid, discogs_id)
data_pool = pd.read_csv(data_pool_file)

# Load dev_set.json (list of dicts, each with "release_group_mbid")
with open(dev_set_file, "r", encoding="utf-8") as f:
    dev_set = json.load(f)

dev_mbids = {entry["release_group_mbid"] for entry in dev_set}

# Filter rows: keep only those not in dev_set
filtered_data = data_pool[~data_pool["mbid"].isin(dev_mbids)]

# Load existing analysis_set.json if exists
if analysis_set_file.exists():
    with open(analysis_set_file, "r", encoding="utf-8") as f:
        try:
            analysis_set = json.load(f)
        except json.JSONDecodeError:
            analysis_set = []
else:
    analysis_set = []

# Build a set of already processed MBIDs to skip
processed_mbids = {entry["release_group_mbid"] for entry in analysis_set if "release_group_mbid" in entry}

# Keep only unprocessed rows
filtered_data = filtered_data[~filtered_data["mbid"].isin(processed_mbids)]

# (Optional) Limit rows for testing
# filtered_data = filtered_data.head(50)

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_row(row):
    mbid = row["mbid"]
    discogs_id = row["discogs_master_id"]

    try:
        result: CoverDataModel = agg.get_rg_cover_data_visual_features(mbid, discogs_id)
        return {"success": True, "data": result.model_dump(mode="json")}
    except Exception as e:
        return {"success": False, "error": str(e), "mbid": mbid, "discogs_id": discogs_id}


# ---- Parallel execution ----
errors_count = 0

batch_size = 100
processed = 0

with ThreadPoolExecutor(max_workers=4) as executor:  # adjust 3–5 depending on stability
    futures = {executor.submit(process_row, row): row for _, row in filtered_data.iterrows()}

    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
        result = future.result()
        if result["success"]:
            analysis_set.append(result["data"])
        else:
            print(f"Error processing mbid={result['mbid']}, discogs_id={result['discogs_id']}: {result['error']}")
            errors_count += 1
        
        processed += 1
        # Save in batches to reduce I/O overhead
        if processed % batch_size == 0:
            with open(analysis_set_file, "w", encoding="utf-8") as f:
                json.dump(analysis_set, f, indent=2, ensure_ascii=False)

print(f"Total errors during the process: {errors_count}/{len(filtered_data)}")
print(f"Final dataset size: {len(analysis_set)}")