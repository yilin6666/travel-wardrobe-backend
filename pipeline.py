import os
import uuid
import json
import base64
import shutil
import tempfile
from query_embedding import call_gemini_api_with_image, call_gemini_api_with_text, validate_person_features, validate_environment_features, flatten_json, clean_and_format
from retrieve import query_index, find_json_data, StorageContext, load_index_from_storage, run_retrieval
from google_image import extract_fields_from_json, build_prompt, model as google_image_model
from google_image import fetch_all_vecteezy_images, get_vecteezy_images_from_environment
from SD import call_gemini as sd_prompt_generator, call_stable_diffusion, get_image_with_r1_sd
from datetime import datetime
from try_on import generate_final_tryon
from config import RESULT_CACHE_PATH
from config import GEMINI_API_KEY

# GOOGLE_API_KEY = "AIzaSyCasR3JT3PbkmMVPuYSoY7G7B-kSkLhQA0"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Persistent result cache
# RESULT_CACHE_PATH = "result_cache.json"

# Load existing cache if available
if os.path.exists(RESULT_CACHE_PATH):
    with open(RESULT_CACHE_PATH, "r") as f:
        results_store = json.load(f)
else:
    results_store = {}

def save_results_cache():
    with open(RESULT_CACHE_PATH, "w") as f:
        json.dump(results_store, f)

def encode_image_url_to_base64(url):
    import requests
    from io import BytesIO
    response = requests.get(url)
    return base64.b64encode(BytesIO(response.content).read()).decode()

def generate_all_results(description, fullbodyshot_path, req_id, selectedImageList=None):
    # Step 1: Generate U_Profile from U2 image
    user_folder = os.path.join("user_images", req_id)
    os.makedirs(user_folder, exist_ok=True)
    # shutil.copy(fullbodyshot_path, os.path.join(user_folder, f"{req_id}_fullbody.jpg"))

    profile_str = call_gemini_api_with_image(fullbodyshot_path)
    person_data = json.loads(profile_str)
    if not validate_person_features(person_data):
        raise ValueError("Missing critical person features: Gender / Age / Skin Tone")

    # Step 2: Generate Text_Q and flatten as Q1
    text_q_str = call_gemini_api_with_text(description)
    scenarios = json.loads(text_q_str)
    print("[Gemini Text Output]", text_q_str)
    if isinstance(scenarios, dict):
        scenarios = [scenarios]

    results = []
    all_q1_texts = []

    clothing_counter = 1  # for unique clothing image names
    tryon_counter = 1     # for unique try-on image names
    background_counter = 1  # for unique background image names

    for i, scenario in enumerate(scenarios):
        if not validate_environment_features(scenario):
            continue

        merged = person_data.copy()
        merged.update(scenario)

        # flatten and save query
        flat = flatten_json(merged)
        q1_text = clean_and_format(flat)
        all_q1_texts.append(q1_text)

        # save temp Q1 txt
        # Q1 path (text file for retrieval input)
        q1_path = f"data_embedding/query_{req_id}.txt"
        with open(q1_path, "w", encoding="utf-8") as f:
            f.write(q1_text)
        
        r1_json_data = run_retrieval(req_id)  # step3: R1 Retrieval from Q1
        r2_images = []
        r3_images = []
        # print(r1_json_data)  # test
        
        for r1 in r1_json_data:
            # Extract environment fields from R1
            env_fields = extract_fields_from_json(r1)

            # Step 4: R2 - Google Image prompt from R1 JSON, Vecteezy image retrieval using Gemini + Environment fields
            vecteezy_results, status = get_vecteezy_images_from_environment(env_fields)

            if status == "Success":
                for vecteezy_url, img_url in vecteezy_results:
                    try:
                        img_base64 = encode_image_url_to_base64(img_url)
                        r2_images.append({
                            "url": img_url,
                            # "base64": img_base64,
                            "source": vecteezy_url,  # optionally store original page
                            "type": f"background_{background_counter}"
                        })
                        background_counter += 1
                    except Exception as e:
                        print(f"Failed to encode image from {img_url}: {e}")
                        continue
            else:
                print(f"Vecteezy retrieval failed for req_id={req_id}: {status}")
            print(r2_images)

            # Step 5: R3 - generate image via SD module
            clothing_image_path = os.path.join("user_images", req_id, f"{req_id}_clothing_{clothing_counter}.png")
            image_path = get_image_with_r1_sd(r1, req_id, clothing_image_path)
            clothing_counter += 1

            if image_path:
                with open(image_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    r3_images.append({
                        "url": f"/user_images/{req_id}/{os.path.basename(image_path)}",
                        # "base64": b64
                    })

        # Try-on generation
        # Get background list from API input (replace with actual passed list in real API call)
        tryon_data = []
        if selectedImageList:
            for bg_url in selectedImageList:
                tryon_path, tryon_b64 = generate_final_tryon(
                    req_id=req_id,
                    clothing_path=image_path,
                    background_url=bg_url,
                    full_body_path=fullbodyshot_path,
                    tryon_counter=tryon_counter
                )
                tryon_counter += 1  # Increment to ensure unique filenames
                if tryon_path and tryon_b64:
                    r3_images.append({
                        "url": tryon_path,
                        # "base64": final_b64,
                        "type": f"final_tryon_{tryon_counter - 1}"
                    })

        # Append try-on images to result3
        r3_images.extend(tryon_data)

        results.append({
            "query1": q1_text,
            "result1": r1_json_data,
            "result2": r2_images,
            "result3": r3_images
        })

    save_results_cache()
    return {
        "id": req_id,
        "profile": fullbodyshot_path,
        "query1": [r["query1"] for r in results],
        "result1": [r["result1"] for r in results],
        "result2": [r["result2"] for r in results],
        "result3": [r["result3"] for r in results]
    }


# test
if __name__ == "__main__":
    test_description = "I am a 28-year-old woman with a slender body, long straight blonde hair, and a warm skin tone. I will be attending my best friend’s wedding in Tuscany in mid-July. I know it will be quite warm and sunny, and the wedding will be outdoors in a vineyard with a rustic setting. The dress code is semi-formal, so I want elegant yet breathable dresses in soft pastel tones, along with accessories like a sun hat and pearl earrings. Please suggest a full outfit and provide images."
    test_image_path = "/Users/elaine/Desktop/fashion_mining/travel-wardrobe-backend/user_images/1.jpg"  # Must exist locally
    test_selected_images = [
    "https://static.vecteezy.com/system/resources/previews/059/892/368/large_2x/scenic-vineyard-landscape-with-lush-rows-of-grapevines-stretching-towards-majestic-mountains-under-a-clear-blue-sky-photo.jpg",
    "https://static.vecteezy.com/system/resources/previews/060/845/738/large_2x/golden-hour-vineyard-landscape-rows-of-grapevines-at-sunset-free-photo.jpg"
]
    result = generate_all_results(test_description, test_image_path, "20250515_173158", selectedImageList=test_selected_images)
    print(json.dumps(result, indent=2))

# 衣服、背景可能产生多个，要按序号命名，不然存储会覆盖
# 衣服和背景的循环不对，背景产生应该是单独的
# 如果用户选择2张以上图片，只返回最后一张图片的结果
# 而try_on 的逻辑要确认下，每个衣服在每个背景上展示
# prompt改一下，横的图片配上全身照，背景变成两张横的图片
  