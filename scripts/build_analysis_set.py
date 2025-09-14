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

# Load dev_set.json (list of dicts, each with "mbid")
with open(dev_set_file, "r", encoding="utf-8") as f:
    dev_set = json.load(f)

dev_mbids = {entry["release_group_mbid"] for entry in dev_set}

# Filter rows: keep only those not in dev_set
filtered_data = data_pool[~data_pool["mbid"].isin(dev_mbids)]

analysis_set = []


# Limit to first 5 rows for testing
filtered_data = filtered_data.head(5)

# Process each row
errors_count = 0
for _, row in tqdm(filtered_data.iterrows(), total=len(filtered_data), desc="Processing"):
    mbid = row["mbid"]
    discogs_id = row["discogs_master_id"]

    try:
        result: CoverDataModel = agg.get_rg_cover_data_visual_features(mbid, discogs_id)
        # Append serialized pydantic model
        analysis_set.append(result.model_dump(mode="json"))
    except Exception as e:
        print(f"Error processing mbid={mbid}, discogs_id={discogs_id}: {e}")
        traceback.print_exc()
        errors_count = errors_count + 1
        continue

    # Save progress incrementally
    with open(analysis_set_file, "w", encoding="utf-8") as f:
        json.dump(analysis_set, f, indent=2, ensure_ascii=False)

print(F"Total errors during the process: {errors_count}")
