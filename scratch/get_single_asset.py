import httpx

asset_id = 709
url_get = f"http://bob.fritz.box:8899/api/audio-assets/{asset_id}"

try:
    r = httpx.get(url_get, timeout=5)
    print("Status:", r.status_code)
    print("Asset:", r.json())
except Exception as e:
    print("Error:", e)
