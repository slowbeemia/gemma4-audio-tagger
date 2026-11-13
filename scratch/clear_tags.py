import httpx

asset_id = 709
url_tag = f"http://bob.fritz.box:8899/api/audio-assets/{asset_id}/tag"
url_untagged = "http://bob.fritz.box:8899/api/audio-assets/untagged"

# Reset asset 709 with Payload 4
print(f"Resetting asset {asset_id} with space tags and empty description...")
try:
    r = httpx.post(url_tag, json={"tags": " ", "description": "", "notes": ""})
    print("Reset status:", r.status_code)
except Exception as e:
    print("Reset error:", e)

# Fetch untagged assets and check if 709 is present
print("Fetching untagged assets...")
try:
    r = httpx.get(url_untagged, params={"limit": 100})
    data = r.json()
    assets = data.get("assets", [])
    found = False
    for a in assets:
        if a["id"] == asset_id:
            print(f"SUCCESS! Asset {asset_id} is in the untagged list! Details: {a}")
            found = True
            break
    if not found:
        print(f"FAILED: Asset {asset_id} was NOT found in the untagged list.")
except Exception as e:
    print("Fetch error:", e)
