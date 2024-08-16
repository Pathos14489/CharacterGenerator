import requests
import json
import time
import random
import os
from tqdm import trange, tqdm
import chromadb
from chromadb.config import Settings
import fastapi
from openai import OpenAI
import llama_cpp
import json
from typing import Annotated, List
from pydantic import BaseModel, Extra, Field
import uuid
import uvicorn
import gradio as gr
import argparse
input_dir_1 = './character_card_v2/'
input_dir_2 = './character_card_v2_lorebooks/'

chroma_path = f"./chromadb"
chroma_client = chromadb.PersistentClient(chroma_path,Settings(anonymized_telemetry=False))

entries = chroma_client.get_or_create_collection(name="entries")

try:
    with open('processed_ids', 'r') as f:
        processed_ids = f.read().splitlines()
except:
    processed_ids = []

# functions to load data from either lorebooks or character_cards

class Character(BaseModel):
    """Character Card V2 Schema"""
    name: str
    description: str
    creator_notes: str
    personality: str
    appearance_description: str
    scenario: str
    tags: list[str]
    first_mes: str

def load_character_book_data(input_dir):
    global processed_ids
    global entries
    print("Loading character book data from", input_dir)
    for filename in tqdm(os.listdir(input_dir)):
        with open(f'{input_dir}{filename}', 'r') as f:
            data = json.load(f)
            if "data" not in data or data["data"] is None:
                continue
            data = data["data"]
            if "character_book" not in data or data["character_book"] is None:
                continue
            try:
                data_id = filename.split('.')[0]
                ids = []
                documents = []
                metadatas = []
                for entry in data['character_book']['entries']:
                    if "content" not in entry or entry["content"] is None or entry["content"].strip() == "" or entry["content"].strip() == "content":
                        continue
                    entry_id = data_id+"_entry_"+str(entry['id'])
                    if entry_id in processed_ids:
                        continue
                    ids.append(entry_id)
                    documents.append(entry["content"])
                    metadata = {
                        # "tags": "|".join(entry['keys']),
                        "name": entry['name'],
                        "probability": entry['probability'],
                        "case_sensitive": entry['case_sensitive'],
                        "constant": entry['constant'],
                        "enabled": entry['enabled'],
                        "priority": entry['priority'],
                        "position": entry['position'],
                        "insertion_order": entry['insertion_order'],
                    }
                    for key in entry['keys']:
                        metadata["tag_"+key] = True
                    metadatas.append(metadata)
                    
                    processed_ids.append(entry_id)
                    with open('processed_ids', 'a') as f:
                        f.write(entry_id + '\n')
                if len(ids) > 0 and len(documents) == len(ids) and len(metadatas) == len(ids):
                    entries.add(ids=ids, documents=documents, metadatas=metadatas)
            except Exception as e:
                print(f"Error processing {filename}\n{e}")
                print(data)
                raise e
            
def load_character_data(input_dir):
    global processed_ids
    global entries
    print("Loading character data from", input_dir)
    for filename in tqdm(os.listdir(input_dir)):
        with open(f'{input_dir}{filename}', 'r') as f:
            entry_id = filename.split('.')[0]
            if entry_id in processed_ids:
                continue
            data = json.load(f)
            try:
                data = data["data"]
            except:
                print(f"Error processing {filename} - no data key")
                continue
            try:
                data_id = filename.split('.')[0]
                ids = []
                documents = []
                metadatas = []
                if "description" not in data or data["description"] is None or data["description"].strip() == "" or data["description"].strip() == "description":
                    continue
                ids.append(data_id)
                documents.append(data["description"])
                metadata = {
                    # "tags": "|".join(data['tags']),
                    "name": data['name'],
                    "first_msg": data['first_mes'],
                    "personality": data['personality'],
                    "post_history_instructions": data['post_history_instructions'],
                    "scenario": data['scenario'],
                    "creator_notes": data['creator_notes'],
                    "creator": data['creator'],
                }
                for tag in data['tags']:
                    metadata["tag_"+tag] = True
                metadatas.append(metadata)
                
                processed_ids.append(data_id)
                with open('processed_ids', 'a') as f:
                    f.write(data_id + '\n')
                if len(ids) > 0 and len(documents) == len(ids) and len(metadatas) == len(ids):
                    entries.add(ids=ids, documents=documents, metadatas=metadatas)
            except Exception as e:
                print(f"Error processing {filename}\n{e}")
                print(data)
                raise e

def entry_search(query_texts, n_results, where=None):
    response = entries.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
    )
    results = []
    for idx, id in enumerate(response['ids'][0]):
        document = response['documents'][0][idx]
        metadata = response['metadatas'][0][idx]
        score = response['distances'][0][idx]
        result = {
            "id": id,
            "score": score,
        }
        if "personality" in metadata:
            result["description"] = document
        else:
            result["content"] = document
        tags = []
        for key in metadata:
            if key.startswith("tag_"):
                tags.append(key[4:])
            else:
                result[key] = metadata[key]
        result["tags"] = tags
        result["tags"] = [tag.strip() for tag in tags]
        if result["name"].strip() == "":
            if len(result["tags"]) > 0:
                result["name"] = result["tags"][0]
        results.append(result)
    return results

def generate_character(query, n_results, where=None, temperature=1.15, min_p=0.05, top_p=1.0):
    global client
    global entries
    query_texts = [query]
    results = entry_search(query_texts=query_texts, n_results=n_results, where=where)
    character_card_schema = Character.model_json_schema()
    print("Using grammar from schema:", json.dumps(character_card_schema, indent=2))
    # grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(character_card_schema))
    messages = [
        {
            "role": "system",
            "content": "You are a character card generator. You will be given a description of a character card to generate. You will then generate a character card that matches the description.\nHere are some related references to use when creating your character card:",
        }
    ]
    for result in results:
        messages.append({
            "role": "system",
            "content": f"Name: {result['name']}\nDescription: {result['description']}\nTags: {', '.join(result['tags'])}\nFirst Message: {result['first_msg']}"
        })
    messages.append({
        "role": "user",
        "content": f"I want a character card made that matches the following description: {query.strip()}",
    })
    # print(messages)
    print("Generating character card...")
    response = client.chat.completions.create(
        model="L3-8B-Stheno-v3.2-Q6_K",
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        extra_body={
            "grammar": character_card_schema,
            "min_p": min_p,
        },
        max_tokens=3072
    )
    raw_character = response.choices[0].message.content
    if "\\\"" in raw_character:
        raw_character = raw_character.replace("\\\"", "\"")
    try:
        character = json.loads(raw_character)
    except Exception as e:
        print("Error parsing response:", e)
        character = response.choices[0].message.content
    return {
        "query": query,
        "character": character,
        "references": results,
    }

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--server_mode", type=str, default="api", choices=["api", "gradio"])
    args.add_argument("--dedupe_on_startup", action="store_true")
    args.add_argument("--load_data", action="store_true")
    args = args.parse_args()
    client = OpenAI(api_key="abc123", base_url="http://localhost:8000/v1/")
    if args.load_data:  
        load_character_book_data(input_dir_1)
        load_character_book_data(input_dir_2)
        load_character_data(input_dir_1)
    if args.dedupe_on_startup:
        # Dedupe
        all_documents = entries.get(include=["documents"])
        print("Loaded", len(all_documents["documents"]), "entries")
        print("Deduping entries")
        documents_seen = set()
        ids_to_remove = []
        with open('delete_ids', 'r') as f:
            for line in f:
                ids_to_remove.append(line.strip())
        for idx, id in enumerate(all_documents["ids"]):
            document = all_documents["documents"][idx]
            if document in documents_seen:
                ids_to_remove.append(id)
            elif len(document) < 5 or len(document.split(" ")) < 5: # Remove entries with very short descriptions, they're probably not useful
                ids_to_remove.append(id)
            else:
                documents_seen.add(document)
        if len(ids_to_remove) > 0:
            if len(ids_to_remove) > 5461:
                batches = [ids_to_remove[i:i + 5461] for i in range(0, len(ids_to_remove), 5461)]
                for batch in batches:
                    print("Deleting", len(batch), "entries")
                    entries.delete(ids=batch)
            else:
                print("Deleting", len(ids_to_remove), "entries")
                entries.delete(ids=ids_to_remove)
        all_documents = entries.get(include=["documents"])
        print("Loaded", len(all_documents["documents"]), "entries")

    if args.server_mode == "api":
        app = fastapi.FastAPI()
        
        @app.post("/search")
        async def search(request: fastapi.Request):
            request = await request.json()
            query = request["query"]
            query_texts = [query]
            n_results = request.get("n_results", 3)
            where = request.get("where", None)
            print("Searching for query:", query)
            results = entry_search(query_texts=query_texts, n_results=n_results, where=where)
            return {
                "results": results,
            }
        @app.post("/generate")
        async def generate(request: fastapi.Request):
            req = await request.json()
            query = req["query"]
            print("Generating character card for query:", query)
            n_results = req.get("n_results", 3)
            query_texts = [query]
            print("Getting relevant references...")
            n_results = req.get("n_results", 3)
            where = req.get("where", None)
            temperature = req.get("temperature", 1.15)
            min_p = req.get("min_p", 0.05)
            top_p = req.get("top_p", 1.0)
            return generate_character(query, n_results, where, temperature, min_p, top_p)

        
        @app.delete("/entry/{entry_id}")
        async def delete_entry(entry_id: str):
            entries.delete(ids=[entry_id])
            return {
                "success": True,
            }
        uvicorn.run(app, host="localhost", port=8024)
    elif args.server_mode == "gradio":
        with gr.Blocks() as mem_gr_blocks:
            gr.Label(value="Character Generator Prototype")
            with gr.Tab(label="Search") as search_tab:
                query = gr.Textbox(lines=5, label="Query")
                n_results = gr.Number(value=3, label="Number of Results")
                only_use_character_data_search = gr.Checkbox(label="Only Use Characters as References")
                search_btn = gr.Button(value="Search")
                search_results = gr.JSON(label="Results")
            with gr.Tab(label="Generate") as generate_tab:
                query_generate = gr.Textbox(lines=5, label="Query")
                n_results_generate = gr.Number(value=3, label="Number of References")
                only_use_character_data_generate = gr.Checkbox(label="Only Use Characters as References")
                temperature = gr.Slider(value=1.15, label="Temperature", minimum=0.0, maximum=3.0)
                min_p = gr.Slider(value=0.05, label="Min P", minimum=0.0, maximum=1.0)
                top_p = gr.Slider(value=1.0, label="Top P", minimum=0.0, maximum=1.0)
                generate_btn = gr.Button(value="Generate")
                generate_results = gr.JSON(label="Results")
            def search(query, n_results, only_use_character_data_search):
                where = None
                if only_use_character_data_search:
                    where = {
                        "first_msg":{
                            "$ne":""
                        }
                    }
                return entry_search([query], n_results, where)
            def generate(query, n_results, only_use_character_data_generate, temperature, min_p, top_p):
                where = None
                if only_use_character_data_generate:
                    where = {
                        "first_msg":{
                            "$ne":""
                        }
                    }
                return generate_character(query, n_results, where, temperature, min_p, top_p)
            search_btn.click(search, [query, n_results, only_use_character_data_search], outputs=search_results)
            generate_btn.click(generate, [query_generate, n_results_generate, only_use_character_data_generate, temperature, min_p, top_p], outputs=generate_results)
            mem_gr_blocks.queue().launch(
                share=True,
            )