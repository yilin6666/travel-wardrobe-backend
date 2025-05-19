import base64
from openai import OpenAI
import os
import requests
from PIL import Image
from io import BytesIO
from config import OPENAI_API_KEY, search_model, TRYON_PROMPT

# 初始化 OpenAI 客户端
client = OpenAI(api_key=OPENAI_API_KEY)
# # Download image 
# def download_and_save_image(url, save_path):
#     try:
#         response = requests.get(url)
#         response.raise_for_status()
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         with open(save_path, "wb") as f:
#             f.write(response.content)
#         print(f"✅ Background image saved: {save_path}")
#         return save_path
#     except Exception as e:
#         print(f"❌ Error downloading image from {url}: {e}")
#         return None

# 构建 Prompt
prompt = TRYON_PROMPT
# prompt = """
# Seamlessly overlay the clothing and accessories from the second reference image onto the person in the main image.
# Replace the original background of the main image with the environment depicted in the third reference image.
# Ensure a highly realistic and photorealistic try-on effect by accurately aligning the clothing with the subject’s posture, body shape, and orientation.
# Maintain the subject’s identity, including facial features, hairstyle, skin tone, and body proportions, without distortion or alteration.
# Harmonize lighting, shadows, and color tones across all image elements to achieve consistent and natural integration.
# Do not modify the facial structure, expression, hairstyle, or any identity-related attributes from the main image. Treat the face area as fixed and immutable.
# Ensure clean edges, proper clothing fitting, and realistic texture rendering of the clothing in the new background scene.
# Ensure the subject’s size and position appear naturally integrated into the new background, with consistent perspective and appropriate scale relative to the environment.
# """
# 注意：目前 OpenAI API 不支持为图片打标签（如 {"person": ..., "clothing": ...}）。所以不能直接在 prompt 中使用图片的“文件名”或“图片名”来区分图像角色
# 建议：为了确保上传顺序一致，建议你在 Web 应用中：明确控制上传顺序（如三个 FormData 字段分别命名为 person, clothing, background）；后端构建 image=[...] 列表时保持固定顺序。

# def download_image_from_url(url, save_path):
#     response = requests.get(url)
#     if response.status_code == 200:
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         with open(save_path, "wb") as f:
#             f.write(response.content)
#         return save_path
#     else:
#         raise Exception(f"Failed to download background image from {url}")

def generate_final_tryon(req_id, clothing_path, background_url, full_body_path, tryon_counter=1):
    # Create output path
    root_id = req_id.split("_")[0]
    out_dir = os.path.join("user_images", root_id)
    os.makedirs(out_dir, exist_ok=True)

    # File paths
    out_path = os.path.join(out_dir, f"{root_id}_tryon_{tryon_counter}.png")
    clothing_path = os.path.join("user_images", req_id, os.path.basename(clothing_path))
    full_body_path = os.path.join("user_images", root_id, f"{root_id}_fullbody.jpg")
    background_path = os.path.join(out_dir, f"{root_id}_background_{tryon_counter}.jpg")

    # Download background
    if not os.path.exists(background_path):
        try:
            response = requests.get(background_url)
            response.raise_for_status()
            with open(background_path, "wb") as f:
                f.write(response.content)
            print(f"✅ Background image saved: {background_path}")
        except Exception as e:
            print(f"❌ Error downloading image from {background_url}: {e}")
            return "", ""
        
    try:
        # Open image files in binary mode
        with open(full_body_path, "rb") as f1, open(clothing_path, "rb") as f2, open(background_path, "rb") as f3:
            result = client.images.edit(
                model=search_model,
                image=[f1, f2, f3],
                prompt=prompt
            )

        image_data = base64.b64decode(result.data[0].b64_json)
        with open(out_path, "wb") as f:
            f.write(image_data)

        print(f"✅ Try-on image saved to {out_path}")
        return out_path, base64.b64encode(image_data).decode("utf-8")

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return "", ""
    
    # with open(full_body_path, "rb") as f1, open(clothing_path, "rb") as f2, open(background_path, "rb") as f3:
    # You can test this manually like below (example only)
# if __name__ == "__main__":
#     result_path, b64 = generate_final_tryon(
#         req_id="20250515_173158",
#         clothing_path="user_images/20250515_173158/20250515_173158_clothing.png",
#         background_url="https://static.vecteezy.com/system/resources/previews/059/892/368/large_2x/scenic-vineyard-landscape-with-lush-rows-of-grapevines-stretching-towards-majestic-mountains-under-a-clear-blue-sky-photo.jpg",
#         full_body_path="user_images/20250515_173158/20250515_173158_fullbody.jpg"
#     )
#     print("Output Path:", result_path)