import os
import json
import re
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import urllib.parse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import time
from config import GEMINI_API_KEY, image_search_prompt

# 配置API
# GOOGLE_API_KEY = "AIzaSyCasR3JT3PbkmMVPuYSoY7G7B-kSkLhQA0"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")



# 提取字段函数
# def extract_fields_from_json(file_path):  # 原来从文件路径打开
#     extracted = {}
#     with open(file_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)
#     env_data = data.get("Environment-Related Features", {})
#     for field in target_fields:
#         for key in env_data.keys():
#             if key.lower().strip() == field.lower().strip():
#                 value = env_data[key]
#                 extracted[field] = ", ".join(value) if isinstance(value, list) else value
#                 break
#     return extracted

def extract_fields_from_json(data_dict):    # change to accept from r1
    # 抽取字段名
    target_fields = [
        "Season", "Weather", "Time Of Day", "Lighting Style",
        "Location", "Temperature", "Scene Environment",
        "Scene Type", "Scene Features", "Ambience"
    ]
    extracted = {}
    env_data = data_dict.get("Environment-Related Features", {})
    for field in target_fields:
        for key in env_data.keys():
            if key.lower().strip() == field.lower().strip():
                value = env_data[key]
                extracted[field] = ", ".join(value) if isinstance(value, list) else value
                break
    return extracted

# Gemini Prompt
def build_prompt(field_dict):
    prompt = image_search_prompt.format(field_dict=field_dict)
#     prompt = f"""
# Your task is to act as a Google Image Search assistant.  
# You will be given a set of environmental attributes extracted from a text file.  
# Your job is to generate a natural-language search query that can be used in Google Image to retrieve a real-life photo background that fits the described setting.

# The image **must be a real photograph**, not an illustration or AI-generated image.  
# It should reflect the combined environmental context provided.  
# Assume the image is to be used as a realistic background for a fashion recommendation system.

# Fields may include:
# - Season (e.g., Summer, Winter)
# - Weather (e.g., Sunny, Rainy, Snowy)
# - Time of Day (e.g., Morning, Afternoon, Evening)
# - Lighting Style (e.g., Natural light, Direct light)
# - Location (e.g., Beach, Urban street, Lakeside)
# - Temperature (e.g., 20-25°C, Hot, Cold)
# - Scene Environment (e.g., Outdoor, Indoor)
# - Scene Type (e.g., Natural landscape, Street)
# - Scene Features (e.g., Rocks, Trees, Buildings)
# - Ambience (e.g., Calm, Joyful, Dramatic)

# Please follow these instructions:

# 1. Combine all available fields into a **single descriptive sentence** that forms a realistic and precise Google Image search query.
# 2. Ensure the description is **natural and specific enough** to retrieve relevant results (e.g., "sunny beach in summer with clear blue sky and rocks").
# 3. Do **not invent any missing fields**—only use the ones provided.
# 4. Do **not generate or describe an AI-generated image**—your goal is to help retrieve **real-world photos** only.

# Output format:
# Search Query: <your search query>

# Now generate the query for the following fields:
# {field_dict}
# """
    return prompt


def fetch_all_vecteezy_images(search_query):
    from bs4 import BeautifulSoup

    search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote_plus(search_query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    }

    try:
        response = requests.get(search_url, headers=headers)
        if response.status_code != 200:
            return "Search failed", []

        soup = BeautifulSoup(response.text, "html.parser")
        links = [a['href'] for a in soup.find_all('a', href=True)]
        vecteezy_links = [link for link in links if "vecteezy.com/photo/" in link]
        # print("[Debug] All hrefs from Google Image search:")
        # for link in links:
        #     print(link)


        if not vecteezy_links:
            return "No Vecteezy links found", []

        # 初始化浏览器
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f'user-agent={headers["User-Agent"]}')
        driver = webdriver.Chrome(options=chrome_options)

        results = []
        for vecteezy_url in vecteezy_links:
            try:
                driver.get(vecteezy_url)
                time.sleep(2)
                soup_vecteezy = BeautifulSoup(driver.page_source, "html.parser")
                meta_img = soup_vecteezy.find("meta", property="og:image")
                image_url = meta_img["content"] if meta_img else "Image URL not found"
                results.append((vecteezy_url, image_url))
            except Exception as inner_e:
                results.append((vecteezy_url, f"Error: {str(inner_e)}"))

        driver.quit()
        return "Success", results

    except Exception as e:
        return "Error", str(e)


def get_vecteezy_images_from_environment(env_fields: dict):
# Step 1: Build Gemini prompt
    prompt = build_prompt(field_dict=env_fields)

    try:
        # Step 2: Query Gemini for search query
        response = model.generate_content(prompt)
        time.sleep(1)  # Delay for stability
        response_text = response.text.strip()

        if "Search Query:" not in response_text:
            print("Gemini response missing 'Search Query:'")
            return [], "No query found"

        # Step 3: Extract actual search query string
        query = response_text.split("Search Query:")[-1].strip()
        print(f"Extracted Query: {query}")

        # Step 4: Fetch Vecteezy image results
        status, results = fetch_all_vecteezy_images(query)
        if status != "Success":
            print(f"Vecteezy fetch failed: {status}")
            return [], status

        return results, "Success"

    except Exception as e:
        print(f"Error during query/image fetch: {e}")
        return [], "Exception occurred"

# # test
# if __name__ == "__main__":
#     example_fields = {
#         "Weather": "Sunny",
#         "Season": "Spring",
#         "Lighting style": "Natural light",
#         "Scene Environment": "Outdoor",
#         "Scene Type": "Natural landscape",
#         "Scene Features": "Cherry blossom trees, benches",
#         "Ambience": "Serene, peaceful, vibrant",
#         "Temperature": "15-20"
#     }

#     results, status = get_vecteezy_images_from_environment(example_fields)

#     if status == "Success":
#         for i, (page_url, img_url) in enumerate(results, 1):
#             print(f"[{i}] Page: {page_url}")
#             print(f"    Image: {img_url}")
#     else:
#         print("⚠️ Failed:", status)


# # Main
# folder_path = r"C:\Users\92521\Desktop\result"

# print("\n==========================\n")

# for filename in os.listdir(folder_path):
#     if filename.endswith(".json"):
#         file_path = os.path.join(folder_path, filename)
#         fields = extract_fields_from_json(file_path)
#         if not fields:
#             print(f"{filename}: No relevant fields found.\n")
#             continue

#         # 生成查询语句
#         prompt = build_prompt(fields)
#         response = model.generate_content(prompt)
#         response_text = response.text.strip()

#         # 提取查询词
#         search_query = None
#         if "Search Query:" in response_text:
#             search_query = response_text.split("Search Query:")[-1].strip()
#             status, results = fetch_all_vecteezy_images(search_query)

#         else:
#             search_query = "Search query not generated"
#             vecteezy_url = "N/A"

#         # 打印结果
#         print(f"File: {filename}")
#         print(f"Search Query: {search_query}")
#         if status != "Success":
#             print(f"Fetch Status: {status}")
#         else:
#             for i, (page_url, img_url) in enumerate(results, 1):
#                 print(f"[{i}] Vecteezy Page URL: {page_url}")
#                 print(f"    Image URL: {img_url}")
#         print("-" * 70)