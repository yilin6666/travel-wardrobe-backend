import os
import re
import json
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from config import GEMINI_API_KEY, raw_dict, index, embed_model, llm_model, merged_dict

# GOOGLE_API_KEY = "AIzaSyCasR3JT3PbkmMVPuYSoY7G7B-kSkLhQA0"
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# CONFIG
# load_from_existing_embedding_file = True 
# data_folder = "retrieval_txt"
# index_path = "data_embedding"  
# merged_json_path = "/Users/elaine/Desktop/fashion_mining/travel-wardrobe-backend/merged_dict.json"  

# # Load merged JSON once
# with open(merged_json_path, "r", encoding="utf-8") as f:
#     raw_dict = json.load(f)

# Fix keys: remove "b'" prefix and "'" suffix if present
merged_dict = {}
for raw_key, value in raw_dict.items():
    normalized_key = raw_key.strip().strip("b'").strip("'").strip('"')
    merged_dict[normalized_key] = value


def find_json_data(source):
    if source.endswith(".json"):
        source = source[:-5]
    return merged_dict.get(source)

def load_query_from_txt(query_path):
    with open(query_path, 'r', encoding='utf-8') as f:
        return f.read().strip()
    
def query_index(index, query, k=3):
    print("[query_index] Using query:", query)
    llm = GoogleGenAI(model=llm_model, api_key=GEMINI_API_KEY)
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(query)
    results = []
    for node_with_score in response.source_nodes[:k]:
        source = node_with_score.node.metadata.get("source", None)
        score = node_with_score.score
        print(f"[Match] Score: {score:.4f} | Source: {source}")  # print match info

        results.append({
            'text': node_with_score.node.text,
            'score': score,
            'source': source
        })
    return results

# Index builder
def load_and_split_by_marker(file_path, marker="###"):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chunks = [chunk.strip() for chunk in content.split(marker) if chunk.strip()]
    documents = []
    for chunk in chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        file_id = lines[0].strip()
        content_text = "\n".join(lines[1:]).strip()
        documents.append(Document(text=content_text, metadata={"source": file_id}))
    return documents

def load_all_txt_documents(folder_path, marker="###"):
    all_documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            full_path = os.path.join(folder_path, filename)
            chunks = load_and_split_by_marker(full_path, marker)
            all_documents.extend(chunks)
    return all_documents

# pipeline.py call
def run_retrieval(req_id, k=3):
    query_path = f"data_embedding/query_{req_id}.txt"
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found: {query_path}")

    print(f"\n📥 Running retrieval for req_id: {req_id}")
    query = load_query_from_txt(query_path)

    # # Load embedding model + index
    # Settings.embed_model = GoogleGenAIEmbedding(model_name=embed_model, embed_batch_size=100)   
    # Settings.llm = GoogleGenAI(model=llm_model, api_key=GOOGLE_API_KEY)

    # storage_context = StorageContext.from_defaults(
    #     docstore=SimpleDocumentStore.from_persist_dir(index_path),
    #     vector_store=SimpleVectorStore.from_persist_dir(index_path),
    #     index_store=SimpleIndexStore.from_persist_dir(index_path)
    # )
    # index = load_index_from_storage(storage_context)

    # Run query
    results = query_index(index, query, k=k)

    matched_jsons = []
    for i, res in enumerate(results):
        source = res.get("source")

        # 🧹 Normalize: remove ".json" suffix and surrounding quotes if any
        if source:
            source = source.strip().replace(".json", "").strip("'").strip('"')
        else:
            # Try fallback from text using regex
            match = re.search(r"b'([0-9a-f]{32})'", res["text"])
            if match:
                source = match.group(1)
            else:
                print(f"[Result {i+1}] ❌ Cannot extract JSON key")
                continue
        json_data = merged_dict.get(source)
        if json_data:
            matched_jsons.append(json_data)
            print(f"[Result {i+1}] ✅ Matched source: {source} (Score: {res['score']:.4f})")
        else:
            print(f"[Result {i+1}] ⚠️ No JSON found for source: {source}")

    return matched_jsons


if __name__ == "__main__":

    # Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", embed_batch_size=100)
    # Settings.llm = GoogleGenAI(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)

    # if not load_from_existing_embedding_file:
    #     print("加载并嵌入全部 .txt 文档 ...")
    #     documents = load_all_txt_documents(data_folder)
    #     parser = SimpleNodeParser.from_defaults()
    #     nodes = parser.get_nodes_from_documents(documents)
    #     storage_context = StorageContext.from_defaults()
    #     storage_context.docstore.add_documents(nodes)
    #     index = VectorStoreIndex(nodes, storage_context=storage_context)
    #     index.storage_context.persist(persist_dir=index_path)
    #     print(f"索引已保存到：{index_path}")
    # else:
    #     print("Loading index from disk...")
    #     storage_context = StorageContext.from_defaults(
    #         docstore=SimpleDocumentStore.from_persist_dir(index_path),
    #         vector_store=SimpleVectorStore.from_persist_dir(index_path),
    #         index_store=SimpleIndexStore.from_persist_dir(index_path)
    #     )
    #     index = load_index_from_storage(storage_context)
    #     print(f"索引加载成功：{index_path}")   
    
    # test_req_id = "20250515_173158"
    # test_query_path = f"data_embedding/query_{test_req_id}.txt"
    
    # if os.path.exists(test_query_path):
    #     print(f"\n[Query File] {test_query_path}")
    #     query = load_query_from_txt(test_query_path)
    #     results = query_index(index, query, k=3)
    #     for i, res in enumerate(results):
    #         source = res.get('source')

    #         if not source or not isinstance(source, str) or "json" not in source:
    #             match = re.search(r"(b'[0-9a-f]{32}'\.json)", res["text"])
    #             if match:
    #                 source = match.group(1)
    #             else:
    #                 print(f"[Result {i+1}]无法确定 JSON 文件名")
    #                 continue
    #         else:
    #             source = source.strip()

    #         json_data = find_json_data(source)

    #         print(f"[Result {i+1}] (Score: {res['score']:.4f})")
    #         print(f"JSON 文件名: {source}")
    #         if json_data:
    #             print(f"JSON 文件路径: {json_data}")
    #         else:
    #             print("未在指定路径下找到该 JSON 文件")
    #         print("-" * 60)
    test_req_id = "20250515_173158"  # Change to a valid req_id (matches your query txt name)
    results = run_retrieval(test_req_id)

    print("\n🎯 Retrieved JSON results:")
    for i, r1 in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(json.dumps(r1, indent=2))

#     test_query = """Gender: Female
# Age: Young adult
# Skin Tone: Fair
# Hairstyle Hair Color: Black
# Hairstyle Hair Type: Wavy
# Hairstyle Hair Length: Medium
# Hairstyle Specific Hairstyle: Loose
# Pose: Standing
# Face Shape: Oval
# Body Shape: X
# Clothing Fashion Style: Vacation
# Season: Spring
# Weather: Sunny
# Time of Day: Morning
# Lighting style: Natural light
# Location: Urban setting, park
# Temperature: 15-20
# Scene Environment: Outdoor
# Scene Type: Natural landscape
# Scene Features: Cherry blossom trees
# Ambience: Serene"""  
#     print("\n🧪 Running test query...\n")
#     query_index(index, test_query)








