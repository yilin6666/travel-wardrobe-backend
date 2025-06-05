import requests

submit_payload = {
    "headshot": "https://example.com/head.jpg",
    "fullbodyshot": "https://example.com/full.jpg",
    "description": "Chic winter look with trench coat"
}
submit_response = requests.post("http://127.0.0.1:5000/admin/submit", json=submit_payload)
data = submit_response.json()["data"]
print("Submitted ID:", data["id"])

get_results_payload = {
    "id": data["id"],
    "selectedImageList": ["bg1.jpg", "bg2.jpg"]
}
get_response = requests.get("http://127.0.0.1:5000/admin/getResults", params=get_results_payload)
print("Get Results:", get_response.json())