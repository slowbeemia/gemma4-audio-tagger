from google import genai
from google.genai import types

LITELLM_PROXY_URL = "http://localhost:9379"
client = genai.Client(
    api_key="local-proxy-key",
    http_options=types.HttpOptions(
        base_url=LITELLM_PROXY_URL,
        headers={"x-litellm-disable-callbacks": "true"}
    )
)

print("Sending request to LiteLLM without trailing slash...")
try:
    response = client.models.generate_content(
        model="gemma4:12b",
        contents="Say hello!",
    )
    print("Response:", response.text)
except Exception as e:
    print("Error:", e)
