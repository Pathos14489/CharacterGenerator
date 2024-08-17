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
generated_characters = chroma_client.get_or_create_collection(name="generated_characters")
generated_lorebook_entries = chroma_client.get_or_create_collection(name="generated_lorebook_entries")

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
    first_msg: str


class StattedCharacter(BaseModel):
    """Statted Character Card V2 Schema - Uses SPECIAL Stats, 1-10"""
    name: str
    description: str
    creator_notes: str
    personality: str
    appearance_description: str
    scenario: str
    tags: list[str]
    first_msg: str
    strength: int = Field(..., ge=1, le=10)
    perception: int = Field(..., ge=1, le=10)
    endurance: int = Field(..., ge=1, le=10)
    charisma: int = Field(..., ge=1, le=10)
    intelligence: int = Field(..., ge=1, le=10)
    agility: int = Field(..., ge=1, le=10)
    luck: int = Field(..., ge=1, le=10)

class LorebookEntry(BaseModel):
    """Lorebook Entry V2 Schema"""
    name: str
    content: str
    keys: list[str]

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
                    "first_msg": data['first_msg'],
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

def generated_character_search(query_texts, n_results, where=None):
    response = generated_characters.query(
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

def generate_character(query, n_results, character_count, where=None, generate_statted=False, temperature=1.15, min_p=0.05, top_p=1.0, max_tokens=3072):
    global client
    global entries
    query_texts = [query]
    results = entry_search(query_texts=query_texts, n_results=n_results, where=where)
    if generate_statted:
        character_card_schema = StattedCharacter.model_json_schema()
    else:
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
        try:
            messages.append({
                "role": "system",
                "content": f"Name: {result['name']}\nDescription: {result['description']}\nPersonality: {result['personality']}\nTags: {', '.join(result['tags'])}\nFirst Message: {result['first_msg']}\nScenario: {result['scenario']}\nCreator Notes: {result['creator_notes']}"
            })
        except:
            try:
                messages.append({
                    "role": "system",
                    "content": f"Name: {result['name']}\nDescription: {result['content']}\nTags: {', '.join(result['tags'])}"
                })
            except:
                print("Error processing result:", result)
    messages.append({
        "role": "user",
        "content": f"I want a character card made that matches the following description: {query.strip()}",
    })
    # print(messages)
    characters = []
    for i in trange(character_count):
        print("Generating character", i+1, "of", character_count)
        response = client.chat.completions.create(
            model="L3-8B-Stheno-v3.2-Q6_K",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "grammar": character_card_schema,
                "min_p": min_p,
            },
            max_tokens=max_tokens
        )
        raw_character = response.choices[0].message.content
        if "\\\"" in raw_character:
            raw_character = raw_character.replace("\\\"", "\"")
        try:
            character = json.loads(raw_character)
        except Exception as e:
            print("Error parsing response:", e)
            character = response.choices[0].message.content
        characters.append(character)
    print("Generated characters...")
    id = str(uuid.uuid4())
    output = {
        "query": query,
        "characters": characters,
        "generation_references": results,
    }
    # with open(f'./requests_log/{id}.json', 'w') as f:
    #     json.dump(output, f, indent=2)
    for idx, character in enumerate(characters):
        if type(character) == str:
            print("There was an error generating a valid character JSON, JSON cannot be parsed:", character)
            continue
        character["id"] = id + "_character_" + str(idx)
        print("Generated character", idx, "with id:", character["id"])
        character["query"] = query
        metadata = {
            "query": query,
            "personality": character["personality"],
            "name": character["name"],
            "scenario": character["scenario"],
            "creator_notes": character["creator_notes"],
            "first_msg": character["first_msg"]
        }
        tags = character["tags"]
        for tag in tags:
            metadata["tag_"+tag] = True
        generated_characters.add(ids=[character["id"]], documents=[character["description"]], metadatas=[metadata])
    return output

def generate_lorebook_entry(query, n_results, entry_count, where=None, temperature=1.15, min_p=0.05, top_p=1.0, max_tokens=3072):
    global client
    global entries
    query_texts = [query]
    results = entry_search(query_texts=query_texts, n_results=n_results, where=where)
    lorebook_entry_schema = LorebookEntry.model_json_schema()
    print("Using grammar from schema:", json.dumps(lorebook_entry_schema, indent=2))
    # grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(lorebook_entry_schema))
    messages = [
        {
            "role": "system",
            "content": "You are a lorebook entry generator. You will be given a description of a lorebook entry to generate. You will then generate a lorebook entry that matches the description.\nHere are some related references to use when creating your lorebook entry:",
        }
    ]
    for result in results:
        messages.append({
            "role": "system",
            "content": f"Name: {result['name']}\nContent: {result['content']}\nTags: {', '.join(result['tags'])}"
        })
    messages.append({
        "role": "user",
        "content": f"I want a lorebook entry made that matches the following description: {query.strip()}",
    })
    # print(messages)
    lorebook_entries = []
    for i in trange(entry_count):
        print("Generating lorebook entry", i+1, "of", entry_count)
        response = client.chat.completions.create(
            model="L3-8B-Stheno-v3.2-Q6_K",
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            extra_body={
                "grammar": lorebook_entry_schema,
                "min_p": min_p,
            },
            max_tokens=max_tokens
        )
        raw_lorebook_entry = response.choices[0].message.content
        if "\\\"" in raw_lorebook_entry:
            raw_lorebook_entry = raw_lorebook_entry.replace("\\\"", "\"")
        try:
            lorebook_entry = json.loads(raw_lorebook_entry)
        except Exception as e:
            print("Error parsing response:", e)
            lorebook_entry = response.choices[0].message.content
        lorebook_entries.append(lorebook_entry)
    print("Generated lorebook entries...")
    id = str(uuid.uuid4())
    output = {
        "query": query,
        "lorebook_entries": lorebook_entries,
        "generation_references": results,
    }
    # with open(f'./requests_log/{id}.json', 'w') as f:
    #     json.dump(output, f, indent=2)
    for idx, lorebook_entry in enumerate(lorebook_entries):
        if type(lorebook_entry) == str:
            print("There was an error generating a valid lorebook entry JSON, JSON cannot be parsed:", lorebook_entry)
            continue
        lorebook_entry["id"] = id + "_lorebook_entry_" + str(idx)
        print("Generated lorebook entry", idx, "with id:", lorebook_entry["id"])
        lorebook_entry["query"] = query
        metadata = {
            "query": query,
            "name": lorebook_entry["name"],
        }
        tags = lorebook_entry["tags"]
        for tag in tags:
            metadata["tag_"+tag] = True
        generated_lorebook_entries.add(ids=[lorebook_entry["id"]], documents=[lorebook_entry["content"]], metadatas=[metadata])
    return output

if __name__ == '__main__':
    import threading
    args = argparse.ArgumentParser()
    args.add_argument("--dedupe_on_startup", action="store_true")
    args.add_argument("--load_data", action="store_true")
    args.add_argument("--share_gradio", action="store_true")
    args.add_argument("--api_hostname", type=str, default="localhost")
    args.add_argument("--api_port", type=int, default=8024)
    args.add_argument("--gradio_hostname", type=str, default="localhost")
    args.add_argument("--gradio_port", type=int, default=8025)
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
    
    @app.post("/search_generated_characters")
    async def search_generated(request: fastapi.Request):
        request = await request.json()
        query = request["query"]
        query_texts = [query]
        n_results = request.get("n_results", 3)
        where = request.get("where", None)
        print("Searching for query:", query)
        results = generated_character_search(query_texts=query_texts, n_results=n_results, where=where)
        return {
            "results": results,
        }
    
    @app.post("/generate_character")
    async def api_generate_character(request: fastapi.Request):
        req = await request.json()
        query = req["query"]
        print("Generating character card for query:", query)
        n_results = req.get("n_results", 3)
        print("Getting relevant references...")
        n_results = req.get("n_results", 3)
        where = req.get("where", None) # {
        #     "first_msg":{
        #         "$ne":""
        #     }
        # }
        temperature = req.get("temperature", 1.15)
        min_p = req.get("min_p", 0.05)
        top_p = req.get("top_p", 1.0)
        character_count = req.get("character_count", 1)
        max_tokens = req.get("max_tokens", 3072)
        return generate_character(query, n_results, where, character_count, temperature, min_p, top_p, max_tokens)
    
    @app.post("/generate_lorebook_entry")
    async def api_generate_lorebook_entry(request: fastapi.Request):
        req = await request.json()
        query = req["query"]
        print("Generating lorebook entry for query:", query)
        n_results = req.get("n_results", 3)
        print("Getting relevant references...")
        n_results = req.get("n_results", 3)
        where = req.get("where", None)
        temperature = req.get("temperature", 1.15)
        min_p = req.get("min_p", 0.05)
        top_p = req.get("top_p", 1.0)
        entry_count = req.get("entry_count", 1)
        max_tokens = req.get("max_tokens", 3072)
        return generate_lorebook_entry(query, n_results, where, entry_count, temperature, min_p, top_p, max_tokens)

    @app.delete("/entry/{entry_id}")
    async def delete_entry(entry_id: str):
        entries.delete(ids=[entry_id])
        return {
            "success": True,
        }
    
    @app.delete("/generated_character/{entry_id}")
    async def delete_generated_character(entry_id: str):
        generated_characters.delete(ids=[entry_id])
        return {
            "success": True,
        }
    
    with gr.Blocks() as mem_gr_blocks:
        gr.Label(value="Character Generator Prototype")
        with gr.Tab(label="Search Reference Material") as search_tab:
            query = gr.Textbox(lines=5, label="Prompt")
            n_results = gr.Number(value=3, label="Number of Results")
            only_use_character_data_search = gr.Checkbox(label="Only Use Characters as References")
            search_btn = gr.Button(value="Search")
            search_results = gr.JSON(label="Results")
        with gr.Tab(label="Search Generated Characters") as search_generated_tab:
            generated_query = gr.Textbox(lines=5, label="Prompt")
            n_results_generated = gr.Number(value=3, label="Number of Results")
            only_use_character_data_search_generated = gr.Checkbox(label="Only Use Characters as References")
            search_generated_btn = gr.Button(value="Search")
            search_generated_results = gr.JSON(label="Results")
        with gr.Tab(label="Generate Character") as generate_tab:
            query_generate = gr.Textbox(lines=5, label="Prompt")
            n_results_generate = gr.Number(value=3, label="Number of References")
            character_count = gr.Number(value=1, label="Number of Characters to Generate")
            generate_statted = gr.Checkbox(label="Generate Statted Characters")
            temperature = gr.Slider(value=1.15, label="Temperature", minimum=0.0, maximum=3.0)
            min_p = gr.Slider(value=0.05, label="Min P", minimum=0.0, maximum=1.0)
            top_p = gr.Slider(value=1.0, label="Top P", minimum=0.0, maximum=1.0)
            max_tokens = gr.Slider(value=3072, label="Max Tokens", minimum=256, maximum=4096, step=128)
            generate_btn = gr.Button(value="Generate")
            generate_results = gr.JSON(label="Results")
        with gr.Tab(label="Generate Lorebook Entry") as generate_lorebook_tab:
            query_generate_lorebook = gr.Textbox(lines=5, label="Prompt")
            n_results_generate_lorebook = gr.Number(value=3, label="Number of References")
            entry_count = gr.Number(value=1, label="Number of Entries to Generate")
            temperature_lorebook = gr.Slider(value=1.15, label="Temperature", minimum=0.0, maximum=3.0)
            min_p_lorebook = gr.Slider(value=0.05, label="Min P", minimum=0.0, maximum=1.0)
            top_p_lorebook = gr.Slider(value=1.0, label="Top P", minimum=0.0, maximum=1.0)
            max_tokens_lorebook = gr.Slider(value=3072, label="Max Tokens", minimum=256, maximum=4096, step=128)
            generate_lorebook_btn = gr.Button(value="Generate")
            generate_lorebook_results = gr.JSON(label="Results")
        with gr.Tab(label="Delete Entry") as delete_tab:
            entry_id = gr.Textbox(label="Entry ID")
            delete_btn = gr.Button(value="Delete Entry")
            delete_results = gr.JSON(label="Results")
        with gr.Tab(label="Delete Generated Character") as delete_generated_tab:
            entry_id = gr.Textbox(label="Entry ID")
            delete_btn = gr.Button(value="Delete Entry")
            delete_results = gr.JSON(label="Results")
        def gr_search(query, n_results, only_use_character_data_search):
            where = None
            if only_use_character_data_search:
                where = {
                    "first_msg":{
                        "$ne":""
                    }
                }
            return entry_search([query], n_results, where)
        def search_generated(query, n_results, only_use_character_data_search_generated):
            where = None
            if only_use_character_data_search_generated:
                where = {
                    "first_msg":{
                        "$ne":""
                    }
                }
            return generated_character_search([query], n_results, where)
        def gr_generate_character(query, n_results, character_count, max_tokens, generate_statted, temperature, min_p, top_p):
            return generate_character(query, n_results, character_count, None, generate_statted, temperature, min_p, top_p, max_tokens)
        def gr_generate_lorebook_entry(query, n_results, entry_count, max_tokens, temperature, min_p, top_p):
            return generate_lorebook_entry(query, n_results, entry_count, None, temperature, min_p, top_p, max_tokens)
        def gr_delete_entry(entry_id):
            entries.delete(ids=[entry_id])
            return {
                "success": True,
            }
        def delete_generated_character(entry_id):
            generated_characters.delete(ids=[entry_id])
            return {
                "success": True,
            }
        delete_btn.click(gr_delete_entry, [entry_id], outputs=delete_results)
        delete_btn.click(delete_generated_character, [entry_id], outputs=delete_results)
        search_btn.click(gr_search, [query, n_results, only_use_character_data_search], outputs=search_results)
        search_generated_btn.click(search_generated, [generated_query, n_results_generated, only_use_character_data_search_generated], outputs=search_generated_results)
        generate_btn.click(gr_generate_character, [query_generate, n_results_generate, character_count, max_tokens, generate_statted, temperature, min_p, top_p], outputs=generate_results)
        generate_lorebook_btn.click(gr_generate_lorebook_entry, [query_generate_lorebook, n_results_generate_lorebook, entry_count, max_tokens_lorebook, temperature_lorebook, min_p_lorebook, top_p_lorebook], outputs=generate_lorebook_results)
            
    print("Starting FastAPI server...")
    app_thread = threading.Thread(target=uvicorn.run, args=(app,), kwargs={
        "host": args.api_hostname,
        "port": args.api_port,
    })
    print(f"Running on local URL:  http://{args.api_hostname}:{args.api_port}")
    app_thread.start()
    print("Starting Gradio server...")
    gradio_thread = threading.Thread(target=mem_gr_blocks.queue().launch, kwargs={
        "share": args.share_gradio,
        "server_name": args.gradio_hostname,
        "server_port": args.gradio_port,
        "prevent_thread_lock":True
    })
    gradio_thread.start()
    print("Servers started.")
    app_thread.join()
    gradio_thread.join()