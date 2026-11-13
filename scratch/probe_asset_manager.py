import httpx

for path in ["/api/audio-assets", "/api/audio-assets/tagged", "/api/audio-assets/list"]:
    try:
        resp = httpx.get(f"http://bob.fritz.box:8899{path}")
        print(f"Path {path}: status={resp.status_code}, content_type={resp.headers.get('content-type')}")
        if resp.status_code == 200:
            print("Content:", resp.text[:400])
    except Exception as e:
         print(f"Path {path} error:", e)
