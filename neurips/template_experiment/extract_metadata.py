import os
import requests
import pandas as pd
from tqdm import tqdm
import re

# Create output folder
os.makedirs("met_images", exist_ok=True)

# Step 1: Search for paintings
search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search"
params = {
    "hasImages": "true",
    "q": "painting"
}
search_response = requests.get(search_url, params=params).json()

object_ids = search_response['objectIDs'][:50000]  # Pull more object IDs

print(f"Found {len(object_ids)} objects. Fetching metadata...")

# Step 2: Fetch metadata
metadata = []

for obj_id in tqdm(object_ids):
    obj_url = f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}"
    obj_data = requests.get(obj_url).json()
    
    title = obj_data.get("title", "")
    artist = obj_data.get("artistDisplayName", "")
    classification = obj_data.get("classification", "")
    date_str = obj_data.get("objectDate", "")
    image_url = obj_data.get("primaryImageSmall", "")
    art_movement = obj_data.get("period", "")  # <- This is where art movement comes from

    # Basic checks: require image
    if not image_url:
        continue

    # --- Parse date carefully ---
    year = None
    if date_str:
        # Remove "ca.", "c.", "circa", etc.
        cleaned_date = re.sub(r'[^0-9]', '', date_str)
        if len(cleaned_date) >= 3:
            try:
                year = int(cleaned_date)
            except ValueError:
                year = None

    if year is None:
        continue  # Only keep if year is successfully parsed

    # Download image
    try:
        img_data = requests.get(image_url, timeout=5).content
        img_filename = f"met_images/{obj_id}.jpg"
        with open(img_filename, "wb") as handler:
            handler.write(img_data)
    except:
        continue  # Skip if download failed

    # Save metadata
    metadata.append({
        "objectID": obj_id,
        "title": title,
        "artist": artist,
        "genre": classification,
        "art_movement": art_movement,
        "year_created": year,
        "original_date_str": date_str,
        "image_filename": img_filename
    })

# Step 3: Save metadata to CSV
df = pd.DataFrame(metadata)
df.to_csv("met_paintings_metadata.csv", index=False)

print(f"Saved {len(metadata)} paintings and metadata.")
