import httpx

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Say hello!"
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.0
    }
}

print("Sending text request to LiteLLM Gemini API endpoint...")
try:
    resp = httpx.post(
        "http://127.0.0.1:9379/v1beta/models/gemma4:12b:generateContent",
        json=payload,
        headers={"x-litellm-disable-callbacks": "true"},
        timeout=10.0
    )
    print("Response Status:", resp.status_code)
    print("Response JSON:", resp.json())
except Exception as e:
    print("Error:", e)
