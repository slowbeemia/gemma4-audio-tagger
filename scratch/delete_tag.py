import httpx

asset_id = 709
url_tag = f"http://bob.fritz.box:8899/api/audio-assets/{asset_id}/tag"

print("Trying DELETE request...")
try:
    r = httpx.delete(url_tag)
    print("DELETE status:", r.status_code)
    print("DELETE response:", r.json())
except Exception as e:
    print("DELETE error:", e)
