import httpx

url = "http://bob.fritz.box:8899/api/audio-assets"
try:
    resp = httpx.get(url, params={"limit": 50})
    print("Status:", resp.status_code)
    data = resp.json()
    print("Keys:", data.keys())
    if "assets" in data:
        print("Total assets:", len(data["assets"]))
        if data["assets"]:
            print("First asset:", data["assets"][0])
except Exception as e:
    print("Error:", e)
