import httpx

url_tag = "http://bob.fritz.box:8899/api/audio-assets/709/tag"
url_untagged = "http://bob.fritz.box:8899/api/audio-assets/untagged"

# Get current total
r = httpx.get(url_untagged, params={"limit": 1})
total_before = r.json().get("total")
print("Total before:", total_before)

# Set tags to space and description to empty
httpx.post(url_tag, json={"tags": " ", "description": "", "notes": ""})

# Get new total
r = httpx.get(url_untagged, params={"limit": 1})
total_after = r.json().get("total")
print("Total after setting space tags:", total_after)

# Now set tags to empty and description to space
httpx.post(url_tag, json={"tags": "", "description": " ", "notes": ""})

# Get new total
r = httpx.get(url_untagged, params={"limit": 1})
total_after2 = r.json().get("total")
print("Total after setting space description:", total_after2)
