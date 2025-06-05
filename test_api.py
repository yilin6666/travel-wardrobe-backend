import requests

# Local Flask server URL
url = "http://127.0.0.1:5000/admin/submit"

# Simulated input
payload = {
    "headshot": "https://example.com/head.jpg",
    "fullbodyshot": "https://example.com/full.jpg",
    "description": "Street style with black leather jacket"
}

response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response:", response.json())

