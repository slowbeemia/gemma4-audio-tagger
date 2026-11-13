import json
import httpx
from pathlib import Path

state_file = Path("c:/Users/dirk/AI/gemma4-audio-tagger/.tagger_state.json")
if not state_file.exists():
    print("State file not found")
    sys.exit(1)

with open(state_file, "r") as f:
    state = json.load(f)

processed_ids = state.get("processed", [])
print(f"Found {len(processed_ids)} processed asset IDs to reset.")

url_template = "http://bob.fritz.box:8899/api/audio-assets/{}/tag"

success_count = 0
fail_count = 0

# We will send the space values as it is the only way the API accepts the reset
payload = {
    "tags": " ",
    "description": " ",
    "notes": ""
}

for asset_id in processed_ids:
    url = url_template.format(asset_id)
    try:
        r = httpx.post(url, json=payload, timeout=5)
        if r.status_code == 200:
            success_count += 1
        else:
            print(f"Failed to reset asset {asset_id}: Status {r.status_code}, Response: {r.text}")
            fail_count += 1
    except Exception as e:
        print(f"Error resetting asset {asset_id}: {e}")
        fail_count += 1

print(f"\nReset completed: {success_count} succeeded, {fail_count} failed.")
