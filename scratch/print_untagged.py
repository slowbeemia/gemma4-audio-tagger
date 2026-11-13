import httpx

url = "http://bob.fritz.box:8899/api/audio-assets/untagged"
try:
    resp = httpx.get(url, params={"limit": 1}, timeout=5)
    print("Status:", resp.status_code)
    data = resp.json()
    print("Keys:", data.keys())
    if "assets" in data and data["assets"]:
        print("First untagged asset:", data["assets"][0])
except Exception as e:
    print("Error:", e)
