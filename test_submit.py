import requests

# Submit POST request to /admin/submit
submit_url = "http://127.0.0.1:5050/admin/submit"
submit_payload = {
    "fullbodyshot": "https://images.pexels.com/photos/4708397/pexels-photo-4708397.jpeg",
    "description": (
        "I’m a 22-year-old female with olive skin and long, straight black hair. "
        "I have a diamond face shape and a petite body type. I’m going on a weekend getaway "
        "to a cozy cabin in the woods during autumn. I expect it to be chilly, and the scenery "
        "will have lots of fall colors. Can you recommend a casual yet stylish outfit that’s perfect "
        "for lounging and taking photos? Include layers, accessories, and footwear."
    )
}

submit_response = requests.post(submit_url, json=submit_payload)
print("/admin/submit Response:")
print(submit_response.status_code)
submit_data = submit_response.json()
print(submit_data)

# Extract req_id for next step
req_id = submit_data["data"]["id"]

# Choose one image from result2 as selectedImageList
selected_image = submit_data["data"]["result2"][0]["url"]

# Submit GET request to /admin/getResults
get_results_url = "http://127.0.0.1:5050/admin/getResults"
params = {
    "id": req_id,
    "selectedImageList": [selected_image]
}

get_response = requests.get(get_results_url, params=params)
print("\n /admin/getResults Response:")
print(get_response.status_code)
print(get_response.json())
