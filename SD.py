import os
import json
import requests
from urllib.parse import urlparse, unquote
import google.generativeai as genai
from config import GEMINI_API_KEY, STABILITY_API_KEY, llm_model, FASHION_PROMPT_TEMPLATE, call_stable_diffusion_url

# API Keys
# GEMINI_API_KEY = "AIzaSyCasR3JT3PbkmMVPuYSoY7G7B-kSkLhQA0"
# STABILITY_API_KEY = "sk-4TiyyN64sJm5AlnncsyF5UWwWZ0FiaJd3ZLROr3cEP1ypMoG"
genai.configure(api_key=GEMINI_API_KEY)

# # 输入与输出路径
# input_dir = "data_embedding"
# output_dir = "generated_images"
# os.makedirs(output_dir, exist_ok=True)

# Prompt
# FASHION_PROMPT_TEMPLATE = """
# Your task is to generate a prompt suitable for a Stable Diffusion model.  
# The goal is to generate an image that only includes **clothing and accessories**, without any person, face, body part, or background scene.

# Guidelines:
# - DO NOT mention or imply any human figure, model, or realistic person.
# - DO NOT suggest any environmental scene, context, or mood.
# - The clothing should appear as if laid flat or displayed on an invisible mannequin.
# - Use a solid white background.
# - Focus entirely on clothing design, style, structure, and accessories.

# Convert the following structured clothing and accessory data into a clear, concise, vivid natural language description suitable for image generation:

# {clothing_dict}

# Output format:
# Fashion Prompt: <your single-sentence description>
# """

def call_gemini(clothing_dict):
    prompt = FASHION_PROMPT_TEMPLATE.format(clothing_dict=json.dumps(clothing_dict, indent=2))
    model = genai.GenerativeModel(llm_model)
    response = model.generate_content(prompt)
    if response.text:
        lines = response.text.splitlines()
        for line in lines:
            if line.lower().startswith("fashion prompt:"):
                return line[len("Fashion Prompt:"):].strip()
        return response.text.strip()
    else:
        raise RuntimeError("No response from Gemini")

# def call_stable_diffusion(prompt_text, file_path):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
#     headers = {
#         "authorization": f"Bearer {STABILITY_API_KEY}",
#         "accept": "image/*"
#     }
#     data = {
#         "prompt": prompt_text,
#         "negative_prompt": (
#             "face, people, person, human, model, portrait, crowd, scenery, background, "
#             "room, photo, sky, realistic face, artistic style, cartoon, text, watermark, blurry"
#         ),
#         "output_format": "png"
#     }

#     response = requests.post(url, headers=headers, files={"none": ''}, data=data)

#     if response.status_code == 200:
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
#         print(f"Image saved: {file_path}")
#     else:
#         print(f"Stable Diffusion API failed: {response.status_code}, {response.text}")

# def get_image_with_r1_sd():
#     for filename in os.listdir(input_dir):
#         if filename.endswith(".json"):
#             json_path = os.path.join(input_dir, filename)
#             with open(json_path, "r", encoding="utf-8") as f:
#                 try:
#                     data = json.load(f)
#                     persons = data.get("Persons", [])
#                     if persons and isinstance(persons, list):
#                         person = persons[0]
#                         clothing_info = person.get("Clothing and Accessories Features", {})
#                         if clothing_info:
#                             print(f"Generating prompt for: {filename}")
#                             prompt_text = call_gemini(clothing_info)
#                             print(f"Gemini prompt: {prompt_text}")
#                             call_stable_diffusion(prompt_text, filename)
#                         else:
#                             print(f"Skipped {filename}: No 'Clothing and Accessories Features' found.")
#                     else:
#                         print(f"Skipped {filename}: No 'Persons' list found.")
#                 except Exception as e:
#                     print(f"Error processing {filename}: {e}")



def call_stable_diffusion(prompt_text, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    url = call_stable_diffusion_url
    headers = {
        "authorization": f"Bearer {STABILITY_API_KEY}",
        "accept": "image/*"
    }
    data = {
        "prompt": prompt_text,
        "negative_prompt": (
            "face, people, person, human, model, portrait, crowd, scenery, background, "
            "room, photo, sky, realistic face, artistic style, cartoon, text, watermark, blurry"
        ),
        "output_format": "png"
    }

    response = requests.post(url, headers=headers, files={"none": ''}, data=data)

    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved: {file_path}")
    else:
        print(f"Stable Diffusion API failed: {response.status_code}, {response.text}")

def get_image_with_r1_sd(r1: dict, req_id: str, file_path: str) -> str:
    # Extract clothing info
    clothing_info = r1.get("Persons", [{}])[0].get("Clothing and Accessories Features", {})
    if not clothing_info:
        print("No clothing info found in R1.")
        return ""

    # Generate prompt
    prompt = call_gemini(clothing_info)

    # Define image path
    # img_dir = os.path.join("user_images", req_id)
    # file_path = os.path.join(img_dir, f"{req_id}_clothing.png") # define in pipeline.py

    # Generate image
    call_stable_diffusion(prompt, file_path)

    # Return path if successful
    return file_path if os.path.exists(file_path) else ""


# # test
# # if __name__ == "__main__":
# if __name__ == "__main__":
#     test_r1 = {
#         "Persons": [
#             {
#                 "Clothing and Accessories Features": {
#                     "Clothing Type": "Flowy floral dress",
#                     "Color": "Soft pastel pink",
#                     "Pattern": "Small cherry blossoms",
#                     "Accessories": ["Wide-brim straw hat", "White sandals"]
#                 }
#             }
#         ]
#     }

#     test_req_id = "test_20250515"

#     image_path = get_image_with_r1_sd(test_r1, test_req_id)
#     if image_path:
#         print(f"✅ Image successfully generated at: {image_path}")
#     else:
#         print("❌ Image generation failed.")