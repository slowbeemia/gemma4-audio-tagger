import httpx

url = "http://bob.fritz.box:8899/api/audio-assets/795"
try:
    resp = httpx.get(url, timeout=5)
    print("Status:", resp.status_code)
    print("Response:", resp.text)
except Exception as e:
    print("Error:", e)
