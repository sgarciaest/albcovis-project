import pandas as pd
import json
from pathlib import Path
from urllib.parse import urlparse

from albcovis.utils.img import download_image
from albcovis.settings import settings
from albcovis.services.musicbrainz import pick_caa_url
from albcovis.services.cover_data_aggregator import CoverDataAggregator



agg = CoverDataAggregator()

data_pool = pd.read_csv("./data/source/data_pool_20250813.csv")

# MBIDs to add manually
new_mbids = [
    "21062c6a-bf91-3d5d-866a-7f54f6918861",
    "7337629a-89cd-341d-8449-4c83968f41d1",
    "0da340a0-6ad7-4fc2-a272-6f94393a7831",
    "d32078fa-1be1-314e-ab14-b7a87be59865",
    "2187d248-1a3b-35d0-a4ec-bead586ff547",
    "5912fc58-e625-3959-9ee1-ce359592a44f",
    "c0eb54b5-757a-452a-a248-d7dbdb39cd18",
    "6c9c4985-3628-3070-b956-b538f30c9bea",
    #
    "bce5f949-a724-3d99-895f-d1c4a2662d7d",
    "42352def-1aab-3000-b548-895ebd869cb6",
    "231b6ff2-b493-3f73-947e-4dc9b42c3fe0",
    "9fb293f8-5084-36d5-aec5-f111eb462c57",
    "fc36583b-e0a9-3a21-adf3-b1bda73fb059",
    "ac98f1fe-9afe-39c8-be60-0f6cb2024a55",
    "055be730-dcad-31bf-b550-45ba9c202aa3",
    "7f4792fe-b563-4554-849a-95a89be71f84"
]

# Create a DataFrame for them (with NaN in missing columns)
new_rows_df = pd.DataFrame({
    "mbid": new_mbids,
    "discogs_id": [None] * len(new_mbids),
    "genre": [None] * len(new_mbids),
    "year": [None] * len(new_mbids),
    "decade": [None] * len(new_mbids),
})
# # Append to main dataframe
# data_pool = pd.concat([data_pool, new_rows_df], ignore_index=True)
# data_pool["discogs_id"] = data_pool["discogs_id"].where(pd.notna(data_pool["discogs_id"]), None)



# Stratified sampling
sample = (
    data_pool
    .groupby(["decade", "genre"], group_keys=False)
    .apply(lambda g: g.sample(n=min(len(g), 2)))
    .reset_index(drop=True)
)

# Append to main dataframe
sample = pd.concat([sample, new_rows_df], ignore_index=True)
sample["discogs_id"] = sample["discogs_id"].where(pd.notna(sample["discogs_id"]), None)


save_dir = settings.source_images_dir
# Efficient iteration with itertuples (faster than iterrows)
results = []
total = len(sample)

for i, row in enumerate(sample.itertuples(index=False), start=1):
    print(f"[{i}/{total}] Processing MBID {row.mbid}...")

    try:
        cover_data = agg.get_release_group_cover_data(
            mbid=row.mbid,
            discogs_hint_id=getattr(row, "discogs_id", None) or None
        )
        cover_art = cover_data.cover_art

        url, size_tag = pick_caa_url(cover_art)

        image_id = cover_art.id
        parsed_url = urlparse(url)
        url_name = Path(parsed_url.path).name
        url_ext = Path(url_name).suffix
        ext = url_ext or ".jpg"
        filename = f"{image_id}-{size_tag}{ext}"

        saved_img_path = download_image(url, save_dir, filename)

        results.append(cover_data.model_dump(mode="json"))

    except Exception as e:
        print(f"    [ERROR] Failed to process MBID {row.mbid}: {e}")

# Save final JSON
output_path = Path(settings.source_data_dir) / "dev_set.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(results)} records to {output_path}")