import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
import requests
from io import BytesIO
from PIL import Image
from pipeline import generate_all_results, results_store, save_results_cache
from try_on import generate_final_tryon
import traceback
import base64
from config import OUTPUT_DIR, RESULT_CACHE_PATH, embed_model, llm_model, GEMINI_API_KEY, index_path
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage

app = Flask(__name__)
CORS(app)

# pre load
# Load embedding model + index

#     docstore=SimpleDocumentStore.from_persist_dir(index_path),
#     vector_store=SimpleVectorStore.from_persist_dir(index_path),
#     index_store=SimpleIndexStore.from_persist_dir(index_path)
# )
# index = load_index_from_storage(storage_context)
    
# # load json
# merged_json_path = "merged_json.json"
# with open(merged_json_path, "r", encoding="utf-8") as f:
#     raw_dict = json.load(f)

@app.route('/admin/submit', methods=['POST'])
def submit():
    data = request.get_json()
    fullbodyshot_url = data.get("fullbodyshot")
    description = data.get("description")

    if not (fullbodyshot_url and description):
        return jsonify({"errNo": 1, "errMsg": "Missing fields"}), 400

    try:
        # Generate request ID
        req_id = str(uuid.uuid4())

        # Download image and save locally
        response = requests.get(fullbodyshot_url)
        if response.status_code != 200:
            return jsonify({"errNo": 2, "errMsg": "Failed to download image"}), 400

        image = Image.open(BytesIO(response.content))
        USER_IMAGE_DIR = os.path.join(OUTPUT_DIR, req_id)
        os.makedirs(USER_IMAGE_DIR, exist_ok=True)  
        local_img_path = os.path.join(USER_IMAGE_DIR, f"{req_id}_fullbody.jpg")
        image.save(local_img_path)

        # Run full LLM pipeline
        result = generate_all_results(description, local_img_path, req_id)

        # Save result
        results_store[req_id] = result
        save_results_cache()

        # Format output structure as per requirement
        flattened_result1 = []
        for r1_list in result.get("result1", []):
            flattened_result1.extend(r1_list)

        flattened_result2 = []
        seen_urls = set()
        for r2_list in result.get("result2", []):
            for img in r2_list:
                if img["url"] in seen_urls:
                    continue
                try:
                    response = requests.get(img["url"], timeout=10)
                    response.raise_for_status()
                    img_base64 = base64.b64encode(BytesIO(response.content).read()).decode()
                    flattened_result2.append({
                        "url": img["url"],
                        # "base64": img_base64
                    })
                except Exception as e:
                    print(f"⚠️ Failed to fetch or encode result2 image {img['url']}: {e}")

        flattened_result3 = []
        for r3_list in result.get("result3", []):
            for img in r3_list:
                try:
                    local_path = img["url"].lstrip("/")
                    with open(local_path, "rb") as f:
                        img_base64 = base64.b64encode(f.read()).decode()
                    flattened_result3.append({
                        "url": img["url"],
                        # "base64": img_base64
                    })
                except Exception as e:
                    print(f"⚠️ Failed to load result3 image {img['url']}: {e}")

        return jsonify({
            "errNo": 0,
            "errMsg": "",
            "data": {
                "profile": fullbodyshot_url,
                "query1": result.get("query1", []),
                "result1": flattened_result1,
                "result2": flattened_result2,
                "result3": flattened_result3
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"errNo": 500, "errMsg": str(e)}), 500

@app.route('/admin/getResults', methods=['GET'])
def get_results():
    req_id = request.args.get("id")
    selected = request.args.getlist("selectedImageList")

    result = results_store.get(req_id)
    if result is None:
        return jsonify({"errNo": 4, "errMsg": "Result not found"}), 404

    # Extract existing cloth image
    tryon_data = []
    profile_path = result["profile"]
    result3 = result.get("result3", [])
    existing_tryon_count = sum(
    1 for scenario in result3 for item in scenario if item.get("type", "").startswith("final_tryon_")
)
    tryon_counter = existing_tryon_count + 1

    for scenario_index, scenario_results in enumerate(result.get("result3", [])):
        if not scenario_results:
            continue
        clothing_path = None
        for item in scenario_results:
            url = item.get("url")
            if url and url.endswith(".png") and "clothing" in url:
                clothing_path = url.lstrip("/")
                break
        if not clothing_path:
            continue  # Skip if no valid clothing URL found

        for bg_url in selected:
            tryon_path, tryon_b64 = generate_final_tryon(
                req_id=req_id,
                clothing_path=clothing_path,
                background_url=bg_url,
                full_body_path=profile_path,
                tryon_counter=tryon_counter
            )
            tryon_counter += 1
            if tryon_path and tryon_b64:
                tryon_data.append({
                    "image": tryon_path,
                    # "base64": tryon_b64,
                    "text": f"final_tryon_{tryon_counter - 1}"
                })

    # Update result3 in memory
    for i, new_tryon in enumerate(tryon_data):
        if i < len(result["result3"]):
            result["result3"][i].append(new_tryon)
        else:
            result["result3"].append([new_tryon])

    return jsonify({
        "errNo": 0,
        "errMsg": "",
        "data": {"list": tryon_data}
    })


@app.route('/')
def index():
    return "Flask server is running."

@app.route('/admin/test', methods=['GET'])
def test_connection():
    return "test ok", 200

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)


# test example (input in terminal)
# curl -X POST http://localhost:5050/admin/submit \  -H "Content-Type: application/json" \
#   -d '{
#         "fullbodyshot": "https://images.pexels.com/photos/4708397/pexels-photo-4708397.jpeg",
#         "description": "I am a 28-year-old woman with a slender body, long straight blonde hair, and a warm skin tone. I will be attending my best friend’s wedding in Tuscany in mid-July. I know it will be quite warm and sunny, and the wedding will be outdoors in a vineyard with a rustic setting. The dress code is semi-formal, so I want elegant yet breathable dresses in soft pastel tones, along with accessories like a sun hat and pearl earrings. Please suggest a full outfit and provide images."
#       }'


# curl -G http://localhost:5050/admin/getResults \
# --data-urlencode "id=e488626b-cb89-485a-96d0-f2fe9285b30c" \        # need to replace id using the one generate from /admin/submit
# --data-urlencode "selectedImageList=https://static.vecteezy.com/system/resources/previews/060/845/738/large_2x/golden-hour-vineyard-landscape-rows-of-grapevines-at-sunset-free-photo.jpg" \
# --data-urlencode "selectedImageList=https://static.vecteezy.com/system/resources/previews/059/892/368/large_2x/scenic-vineyard-landscape-with-lush-rows-of-grapevines-stretching-towards-majestic-mountains-under-a-clear-blue-sky-photo.jpg"

