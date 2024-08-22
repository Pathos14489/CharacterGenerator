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
from pydantic import BaseModel, Extra, Field, conlist
import uuid
import uvicorn
import gradio as gr
import argparse
input_dir_1 = './character_card_v2/'
input_dir_2 = './character_card_v2_lorebooks/'
pantella_input_dir = './pantella_characters/'

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

# Character Schemas
class CharacterV2(BaseModel):
    """Character Card V2 Schema - No stats, just descriptions"""
    name: str
    age: int
    description: str
    creator_notes: str
    personality: str
    appearance_description: str
    scenario: str
    tags: list[str]
    first_msg: str

class CharacterV3(BaseModel):
    """Character Card V3 Schema - No stats, just descriptions. The descriptions are intended to be combined into a single description in the final output, so write them such that Personality, Appearance, and Scenario are all separate sections. that should cohesively flow together, seperated by new lines, and not repeat themselves. Drives are what motivates the character, and can be things like "Revenge on the bandits who killed their family" or "To find the lost city of gold". Tags are used to help search for characters, and can be things like "Elf", "Wizard", "Pirate", etc. The voice description is seperate from the other descriptions, and should be able to exist by itself without the other descriptions. It should describe how the character should sound. All descriptions should be at least a paragraph long, and the first message should be at least a sentence long, but longer is not bad. The backstory is the character's history, and should be at least a paragraph long. The current scenario is what the character is currently doing, and should be at least a sentence long. The naked body description is what the character looks like without clothes, and should be at least a paragraph long and explicitly and graphically describe the character's nude body. All fields are required to have a value."""
    full_name: str
    nick_name: str
    age: int
    gender: str = Field(...,examples=["Male","Female"],pattern="^(Male|Female)$")
    race: str
    species: str
    personality_description: str
    outfit_description: str = Field(...,description="A description of the character's outfit. Should be at least a paragraph long.")
    body_description: str = Field(...,description="A description of the character's body not including the chest or genitals. Should be at least a paragraph long and explicitly and graphically describe the character's nude body.")
    bare_chest_description: str = Field(...,description="A description of the character's chest without clothes. Should be at least a paragraph long and explicitly and graphically describe the character's nude chest.")
    genital_description: str = Field(...,description="A description of the character's genitals. Should be at least a paragraph long and explicitly and graphically describe the character's nude genitals.")
    backstory: str = Field(...,description="A description of the character's backstory. Should be at least a paragraph long.")
    current_scenario: str = Field(...,description="A description of what the character is currently doing. Should be at least a sentence long.")
    voice_description: str = Field(...,description="A description of the character's voice. Should be at least a sentence long.")
    drives: list[str] = Field(...,description="A list of things that motivate the character. Should be at least one item long.")
    tags: list[str]
    first_msg: str

class SPECIALCharacter(BaseModel):
    """SPECIAL Character Schema - Uses SPECIAL Stats, 1-10"""
    name: str
    backstory: str
    personality_description: str
    appearance_description: str
    strength: int = Field(..., ge=1, le=10)
    perception: int = Field(..., ge=1, le=10)
    endurance: int = Field(..., ge=1, le=10)
    charisma: int = Field(..., ge=1, le=10)
    intelligence: int = Field(..., ge=1, le=10)
    agility: int = Field(..., ge=1, le=10)
    luck: int = Field(..., ge=1, le=10)

class SkyrimCharacter(BaseModel):
    """Skyrim Character Schema - Uses Skyrim Stats, Stats are all 0-100"""
    name: str
    backstory: str
    personality_description: str
    appearance_description: str
    illusion: int = Field(..., ge=0, le=100)
    conjuration: int = Field(..., ge=0, le=100)
    destruction: int = Field(..., ge=0, le=100)
    restoration: int = Field(..., ge=0, le=100)
    alteration: int = Field(..., ge=0, le=100)
    enchanting: int = Field(..., ge=0, le=100)
    smithing: int = Field(..., ge=0, le=100)
    heavy_armor: int = Field(..., ge=0, le=100)
    block: int = Field(..., ge=0, le=100)
    two_handed: int = Field(..., ge=0, le=100)
    one_handed: int = Field(..., ge=0, le=100)
    archery: int = Field(..., ge=0, le=100)
    light_armor: int = Field(..., ge=0, le=100)
    sneak: int = Field(..., ge=0, le=100)
    lockpicking: int = Field(..., ge=0, le=100)
    pickpocket: int = Field(..., ge=0, le=100)
    speech: int = Field(..., ge=0, le=100)
    alchemy: int = Field(..., ge=0, le=100)

class PantellaCharacter(BaseModel):
    """Skyrim Character Schema - Uses Skyrim Stats, Stats are all 0-100"""
    name: str
    personality_description: str
    backstory: str = Field(...,description="A description of the character's backstory. Should be at least a paragraph long.")
    current_scenario: str = Field(...,description="A description of what the character is currently doing. Should be at least a sentence long.")
    race: str = Field(...,examples=["Argonian","Breton","Dark Elf","High Elf","Imperial","Khajiit","Nord","Orc","Redguard","Wood Elf"],pattern="^(Argonian|Breton|Dark Elf|High Elf|Imperial|Khajiit|Nord|Orc|Redguard|Wood Elf)$")
    species: str = Field(...,examples=["Human","Mer","Argonian","Daedra", "Divine", "Dragon", "Goblin", "Atronach"], pattern="^(Human|Mer|Argonian|Daedra|Divine|Dragon|Goblin|Atronach)$")
    lang_override: str = Field(...,description="The language/accent to use for the voice lines.",examples=["en","es","fr","de","it","ja","ko","pl","pt","ru","zh"],pattern="^(en|es|fr|de|it|ja|ko|pl|pt|ru|zh)$")
    creator_notes: str = Field(...,description="Any notes about the character from the writer.")


RimworldTrait = Annotated[str, Field(pattern="^(Abrasive|Annoying Voice|Ascetic|Asexual|Beautiful|Bisexual|Bloodlust|Body Modder|Body Purist|Brawler|Cannibal|Careful Shooter|Chemical Fascination|Chemical Interest|Creepy Breathing|Depressive|Fast Learner|Fast Walker|Gay|Gourmand|Great Memory|Greedy|Hard Worker|Industrious|Iron-Willed|Jealous|Jogger|Kind|Lazy|Masochist|Misandrist|Misogynist|Nervous|Neurotic|Night Owl|Nimble|Nudist|Optimist|Pessimist|Pretty|Psychically Deaf|Psychically Dull|Psychically Hypersensitive|Psychically Sensitive|Psychopath|Pyromaniac|Quick Sleeper|Sanguine|Sickly|Slothful|Slow Learner|Slowpoke|Staggeringly Ugly|Steadfast|Super-Immune|Teetotaler|Too Smart|Tortured Artist|Tough|Trigger Happy|Ugly|Undergrounder|Very Neurotic|Volatile|Wimp)$", description="Trait Description")]

class RimworldCharacter(BaseModel):
    """Rimworld Character Schema - Uses Rimworld Stats, Stats are all 0-20
Available Traits:
Trait 	Description
Abrasive 	X always says exactly what's on their mind, especially if it's bugging them. That tends to rub people the wrong way.
Annoying Voice 	X's voice has a particularly grating, nasal quality to it, and tends to talk in barked, garbled phrases. This predisposes others to dislike them.
Ascetic 	X has forsaken physical comforts and enjoyments in favor of a simple, pure lifestyle. They will become unhappy if they have a bedroom that's too impressive. They also dislike fancy food and prefer to eat raw. They never judge others by their appearance.
Asexual 	X has no sexual attraction to anyone at all.
Beautiful 	X is exceptionally beautiful, with an exotic-yet-familiar facial structure and an arresting gaze. People are attracted to them before they even open their mouth.
Bisexual 	X is romantically attracted to both men and women.
Bloodlust 	X gets a rush from hurting people, and never minds the sight of blood or death.
Body Modder 	X feels limited in their feeble feeble human body. X often dreams of being enhanced by artificial body parts or xenogenetics.
Body Purist 	X believes the human body is limited for a reason. To them, artificial body parts are unethical and disgusting.
Brawler 	X likes to fight up close and personal. Their accuracy is greatly increased in melee combat, but they'll be very unhappy if asked to carry a ranged weapon.
Cannibal 	X was taught that eating human meat is wrong and horrible. But one time, long ago, they tried it... and they liked it.
Careful Shooter 	X takes more time to aim when shooting. They shoot less often than others, but with more accuracy.
Chemical Fascination 	X is utterly fascinated with chemical sources of enjoyment. Consuming recreational drugs will create a good mood, while abstaining will lead to increasing frustration over time and possibly drug binges. They will ignore directives to not use recreational drugs, and will consume more than a normal person.
Chemical Interest 	X has an unusual interest in chemical sources of enjoyment. Consuming recreational drugs will create a good mood, while abstaining will lead to increasing frustration over time and possible drug binges. They will ignore directives to not use recreational drugs, and will consume more than a normal person.
Creepy Breathing 	X breathes heavily all the time, and sweats constantly. People find it creepy.
Depressive 	X is perennially unhappy. They have trouble sustaining a good mood even when everything is fine.
Fast Learner 	X has a knack for learning. They pick things up much faster than others.
Fast Walker 	X likes to be where they're going. They walk quicker than most people.
Gay 	X is romantically attracted to people of their own gender.
Gourmand 	X's life revolves around food. They get hungry quickly, and if they are in a bad mood, they will often satisfy their-self by eating.
Great Memory 	X has a fantastic memory for detail. They will lose unused skills at half the rate of other people.
Greedy 	X needs a really impressive bedroom. They get a mood loss if they don't get what they want.
Hard Worker 	X is a natural hard worker and will finish tasks faster than most.
Industrious 	X has an easy time staying on-task and focused and gets things done much faster than the average person.
Iron-Willed 	X's will is an iron shield. They keep going through thick and thin when others broke down long before.
Jealous 	For X, it's degrading to have a less impressive bedroom than someone else. X gets a mood loss if any colonist has a more impressive bedroom.
Jogger 	X always moves with a sense of urgency - so much so that others often fail to keep up.
Kind 	X is an exceptionally agreeable and giving person. They never insult others and will sometimes offer kind words to brighten the moods of those around them. They also never judge people by their appearance.
Lazy 	X is a little bit lazy.
Masochist 	For X, there's something exciting about getting hurt. They don't know why, they're just wired differently.
Misandrist 	X really dislikes and distrusts men.
Misogynist 	X really dislikes and distrusts women.
Nervous 	X tends to crack under pressure.
Neurotic 	X likes to have things squared away. They will work harder than most to attain this state of affairs, but their nerves can get the better of them.
Night Owl 	X likes to work at night. They get a mood bonus if awake at night and mood loss if awake during the day. They don't get a mood penalty for being in the dark.
Nimble 	X has remarkable kinesthetic intelligence. They seem to dance around danger with preternatural grace.
Nudist 	X enjoys the feeling of freedom that comes from being nude. They can handle clothing but will be happier without it.
Optimist 	X is naturally optimistic about life. It's hard to get them down.
Pessimist 	X tends to look on the bad side of life.
Pretty 	X has a pretty face, which predisposes people to like them.
Psychically Deaf 	X's mind works on a psychic frequency different from everyone else. They just aren't affected by psychic phenomena.
Psychically Dull 	X's mind is psychically out of tune with others. They aren't as affected by psychic phenomena.
Psychically Hypersensitive 	X's mind is like a psychic tuning fork. They are extremely sensitive to psychic phenomena.
Psychically Sensitive 	X's mind is unusually sensitive to psychic phenomena.
Psychopath 	X has no empathy. The suffering of others doesn't bother them at all. They don't mind if others are butchered, left unburied, imprisoned, or sold to slavery - unless it affects them. X also feels no mood boost from socializing.
Pyromaniac 	X loves fire. They will never extinguish fires, and will occasionally go on random fire-starting sprees. They will be happy around flames, and happier when wielding an incendiary weapon.
Quick Sleeper 	X doesn't need as much sleep as the average person. Whether they're sleeping on a bed or on the ground, they will be fully rested in about two-thirds of the usual time.
Sanguine 	X is just naturally upbeat about their situation, pretty much all the time, no matter what it is.
Sickly 	X has an awful immune system. They get sick more often than usual, frequently with illnesses that nobody in the colony has been afflicted by.
Slothful 	X loves idleness and hates anything productive. They move slowly and rarely stay focused on a task.
Slow Learner 	X is slow on the uptake. They pick things up much slower than others.
Slowpoke 	X is always falling behind the group whenever they go anywhere.
Staggeringly Ugly 	X is staggeringly ugly. Their face looks like a cross between a drawing by an untalented child, a malformed fetus in a jar of formaldehyde, and a piece of modern art. Others must exert a conscious effort to look at them while conversing.
Steadfast 	X is mentally tough and won't break down under stresses that would crack most people.
Super-Immune 	X has a naturally powerful immune system. They will gain immunity much faster than a normal person would and can survive illnesses that would kill others.
Teetotaler 	X abhors the idea of gaining pleasure from chemicals. They strictly avoid alcohol and recreational drugs.
Too Smart 	X is too smart for their own good. They learn everything much faster than everyone but can be quite eccentric.
Tortured Artist 	X feels alienated and misunderstood by other human beings. They will get a constant mood debuff, but gain a chance (50%) to get creativity inspiration after a mental break.
Tough 	X has thick skin, dense flesh, and durable bones. They take much less damage than other people from the same blows. They are extremely hard to kill.
Trigger Happy 	Pew! Pew! Pew! X just likes pulling the trigger. They shoot faster than others, but less accurately.
Ugly 	X is somewhat ugly. This subtly repels others during social interactions.
Undergrounder 	X has no need to experience the outdoors or light. They will never feel cooped up or get cabin fever, no matter how long they stay inside, and are not bothered by darkness.
Very Neurotic 	X feels constantly nervous about everything that has to get done. They will work extremely hard to attain this state of affairs, but their nerves can easily get the better of them.
Volatile 	X is on a hair-trigger all the time. They are the first to break in any tough situation.
Wimp 	X is weak and cowardly. Even a little pain will immobilize them. 
"""
    full_name: str
    nickname: str
    backstory: str
    appearance_description: str
    personality_description: str
    traits: list[RimworldTrait] = Field(...,description="A list of traits that the character has. Should be at least one item long.",min_items=1,max_items=4)
    shooting: int = Field(..., ge=0, le=5)
    melee: int = Field(..., ge=0, le=5)
    construction: int = Field(..., ge=0, le=5)
    mining: int = Field(..., ge=0, le=5)
    cooking: int = Field(..., ge=0, le=5)
    plants: int = Field(..., ge=0, le=5)
    animals: int = Field(..., ge=0, le=5)
    crafting: int = Field(..., ge=0, le=5)
    artistic: int = Field(..., ge=0, le=5)
    medical: int = Field(..., ge=0, le=5)
    social: int = Field(..., ge=0, le=5)
    intellectual: int = Field(..., ge=0, le=5)

class ScenarioCharacter(BaseModel):
    """Scenario Character Schema - Not a character but a situation/scenario."""
    name: str
    description: str = Field(...,description="A description of the scenario. Should be at least a paragraph long.")
    first_msg: str
    creator_notes: str
    tags: list[str]

# Lorebook Entry Schema
class LorebookEntry(BaseModel):
    """Lorebook Entry V2 Schema"""
    name: str
    content: str
    keys: list[str]

# functions to load data from either lorebooks or character_cards
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
            
def load_pantella_character_data(input_dir):
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
                data_id = filename.split('.')[0]
                ids = []
                documents = []
                metadatas = []
                if "bio" not in data or data["bio"] is None or data["bio"].strip() == "":
                    continue
                ids.append(data_id)
                documents.append(data["bio"])
                for key in data:
                    if data[key] is None:
                        data[key] = ""
                metadata = {
                    "name": data['name'],
                    "bio_url": data['bio_url'],
                    "voice_model": data['voice_model'],
                    "skyrim_voice_folder": data['skyrim_voice_folder'],
                    "lang_override": data['lang_override'],
                    "is_generic_npc": data['is_generic_npc'],
                    "race": data["race"],
                    "species": data["species"],
                    "ref_id": data["ref_id"],
                    "base_id": data["base_id"],
                    "creator_notes": data["notes"],
                }
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
        traits = []
        for key in metadata:
            if key.startswith("trait_"):
                traits.append(key[6:])
            else:
                result[key] = metadata[key]
        drives = []
        for key in metadata:
            if key.startswith("drive_"):
                drives.append(key[6:])
            else:
                result[key] = metadata[key]
        if len(tags) > 0:
            result["tags"] = tags
            result["tags"] = [tag.strip() for tag in tags]
        if len(traits) > 0:
            result["traits"] = traits
            result["traits"] = [trait.strip() for trait in traits]
        if len(drives) > 0:
            result["drives"] = drives
            result["drives"] = [drive.strip() for drive in drives]
        if result["name"].strip() == "":
            if "tags" in result and len(result["tags"]) > 0:
                result["name"] = result["tags"][0]
        results.append(result)
    return results

def generate_character(query, n_results, character_count, where=None, generate_type="None", temperature=1.15, min_p=0.05, top_p=1.0, max_tokens=3072):
    global client
    global entries
    query_texts = [query]
    results = entry_search(query_texts=query_texts, n_results=n_results, where=where)
    if generate_type == "SPECIAL":
        character_card_schema = SPECIALCharacter.model_json_schema()
    elif generate_type == "Rimworld":
        character_card_schema = RimworldCharacter.model_json_schema()
    elif generate_type == "Skyrim":
        character_card_schema = SkyrimCharacter.model_json_schema() 
    elif generate_type == "CharacterV3":
        character_card_schema = CharacterV3.model_json_schema() 
    elif generate_type == "Scenario":
        character_card_schema = ScenarioCharacter.model_json_schema()
    elif generate_type == "Pantella":
        character_card_schema = PantellaCharacter.model_json_schema()
    else:
        character_card_schema = CharacterV2.model_json_schema()
    # schema_description = character_card_schema["description"]
    # schema_description = json.dumps(character_card_schema, indent=2)
    schema_description = character_card_schema["description"]
    for key in character_card_schema:
        # print("Key:", key)
        # print("Value:", character_card_schema[key])
        if type(character_card_schema[key]) == dict:
            if "title" in character_card_schema[key] and character_card_schema[key]["title"] is not None:
                description_part = character_card_schema[key]["title"] + ": "
            else:
                description_part = key + ": "
            add_to_description = False
            if "description" in character_card_schema[key] and character_card_schema[key]["description"] is not None and "title" in character_card_schema[key] and character_card_schema[key]["title"] is not None:
                description_part += character_card_schema[key]["description"]
                add_to_description = True
            if "examples" in character_card_schema[key] and character_card_schema[key]["examples"] is not None:
                description_part += "\nExamples: " + ", ".join(character_card_schema[key]["examples"])
                add_to_description = True
            if add_to_description:
                schema_description += "\n" + description_part
    print("Using grammar from schema:", json.dumps(character_card_schema, indent=2))
    # grammar = llama_cpp.LlamaGrammar.from_json_schema(json.dumps(character_card_schema))
    messages = [
        {
            "role": "system",
            "content": "You are a character generator. You will be given a description of a character to generate. You will then generate a character that matches the description.\nHere are some related references to use when creating your character card:",
        },
        {
            "role": "system",
            "content": schema_description
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
            character = json.loads(raw_character.replace("\u00A0",""))
            if type(character) == str:
                character = json.loads(character)
        except Exception as e:
            print("Error parsing response:", e)
            character = response.choices[0].message.content
        characters.append(character)
    print("Generated characters...")
    id = str(uuid.uuid4())
    output = {
        "characters": [],
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
        character["prompt"] = query
        if generate_type == "SPECIAL":
            metadata = {
                "query": query,
                "personality": character["personality"],
                "name": character["name"],
                "scenario": character["scenario"],
                "creator_notes": character["creator_notes"],
                "first_msg": character["first_msg"],
                "strength": character["strength"],
                "perception": character["perception"],
                "endurance": character["endurance"],
                "charisma": character["charisma"],
                "intelligence": character["intelligence"],
                "agility": character["agility"],
                "luck": character["luck"]
            }
        elif generate_type == "Rimworld":
            metadata = {
                "query": query,
                "personality": character["personality_description"],
                "name": character["full_name"],
                "nickname": character["nickname"],
                "appearance_description": character["appearance_description"],
                "backstory": character["backstory"],
                "shooting": character["shooting"],
                "melee": character["melee"],
                "construction": character["construction"],
                "mining": character["mining"],
                "cooking": character["cooking"],
                "plants": character["plants"],
                "animals": character["animals"],
                "crafting": character["crafting"],
                "artistic": character["artistic"],
                "medical": character["medical"],
                "social": character["social"],
                "intellectual": character["intellectual"]
            }
        elif generate_type == "Skyrim":
            metadata = {
                "query": query,
                "personality": character["personality"],
                "name": character["name"],
                "scenario": character["scenario"],
                "creator_notes": character["creator_notes"],
                "first_msg": character["first_msg"],
                "illusion": character["illusion"],
                "conjuration": character["conjuration"],
                "destruction": character["destruction"],
                "restoration": character["restoration"],
                "alteration": character["alteration"],
                "enchanting": character["enchanting"],
                "smithing": character["smithing"],
                "heavy_armor": character["heavy_armor"],
                "block": character["block"],
                "two_handed": character["two_handed"],
                "one_handed": character["one_handed"],
                "archery": character["archery"],
                "light_armor": character["light_armor"],
                "sneak": character["sneak"],
                "lockpicking": character["lockpicking"],
                "pickpocket": character["pickpocket"],
                "speech": character["speech"],
                "alchemy": character["alchemy"]
            }
        elif generate_type == "CharacterV3":
            metadata = {
                "query": query,
                "personality": character["personality_description"],
                "name": character["full_name"],
                "nick_name": character["nick_name"],
                "age": character["age"],
                "race": character["race"],
                "species": character["species"],
                "backstory": character["backstory"],
                "current_scenario": character["current_scenario"],
                "voice_description": character["voice_description"],
                "first_msg": character["first_msg"]
            }
        elif generate_type == "Scenario":
            metadata = {
                "query": query,
                "name": character["name"],
                "first_msg": character["first_msg"]
            }
        elif generate_type == "Pantella":
            metadata = {
                "query": query,
                "name": character["name"],
                "voice_model": "",
                "skyrim_voice_folder": "",
                "bio_url": "", 
                "ref_id": "",
                "base_id": "",
                "lang_override": character["lang_override"],
                "is_generic_npc": False,
                "race": character["race"],
                "species": character["species"],
                "notes": character["creator_notes"]  
            }
        else:
            metadata = {
                "query": query,
                "personality": character["personality"],
                "name": character["name"],
                "scenario": character["scenario"],
                "creator_notes": character["creator_notes"],
                "first_msg": character["first_msg"]
            }
        if "tags" in character:
            tags = character["tags"]
            for tag in tags:
                metadata["tag_"+tag] = True
        if "traits" in character:
            traits = character["traits"]
            for trait in traits:
                metadata["trait_"+trait] = True
        if "drives" in character:
            drives = character["drives"]
            for drive in drives:
                metadata["drive_"+drive] = True
        character_description = ""
        if generate_type == "SPECIAL" or generate_type == "Skyrim":
            character_description = character["backstory"]
        elif generate_type == "CharacterV3":
            character_description = character["backstory"] + "\n " + character["current_scenario"] + "\n " + character["personality_description"] + "\n " + character["outfit_description"] + "\n " + character["body_description"] + "\n " + character["bare_chest_description"] + "\n " + character["genital_description"]
        elif generate_type == "Pantella":
            character_description = character["personality_description"] + "\n " + character["backstory"] + "\n " + character["current_scenario"]
        elif generate_type == "Rimworld":
            character_description = character["backstory"]
            if character["appearance_description"].strip() != "":
                character_description += "\n" + character["appearance_description"]
            if character["personality_description"].strip() != "":
                character_description += "\n" + character["personality_description"]
        else:
            character_description = character["description"]
        generated_characters.add(ids=[character["id"]], documents=[character_description], metadatas=[metadata])
        character["final_description"] = character_description
        output["characters"].append(character)
    output["schema_description"] = schema_description
    return output, schema_description

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
        lorebook_entry["prompt"] = query
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
    args.add_argument("--allow_searching_generated_characters", action="store_true")
    args.add_argument("--allow_deleting_entries", action="store_true")
    args = args.parse_args()
    client = OpenAI(api_key="abc123", base_url="http://localhost:8000/v1/")
    if args.load_data:  
        load_character_book_data(input_dir_1)
        load_character_book_data(input_dir_2)
        load_character_data(input_dir_1)
        load_pantella_character_data(pantella_input_dir)
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
        if args.allow_searching_generated_characters:
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
            generate_type = gr.Dropdown(label="Stat Type", choices=["None", "SPECIAL", "Rimworld","Skyrim","CharacterV3","Scenario","Pantella"])
            temperature = gr.Slider(value=1.15, label="Temperature", minimum=0.0, maximum=3.0)
            min_p = gr.Slider(value=0.05, label="Min P", minimum=0.0, maximum=1.0)
            top_p = gr.Slider(value=1.0, label="Top P", minimum=0.0, maximum=1.0)
            max_tokens = gr.Slider(value=3072, label="Max Tokens", minimum=256, maximum=4096, step=128)
            generate_btn = gr.Button(value="Generate")
            generate_results = gr.JSON(label="Results")
            schema_description = gr.Textbox(label="Schema Description")
            download_btn = gr.Button(value="Download Results")
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
        if args.allow_deleting_entries:
            with gr.Tab(label="Delete Entry") as delete_tab:
                entry_id = gr.Textbox(label="Entry ID")
                delete_entry_btn = gr.Button(value="Delete Entry")
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
        def gr_generate_character(query, n_results, character_count, max_tokens, generate_type, temperature, min_p, top_p):
            results, schema_desc = generate_character(query, n_results, character_count, None, generate_type, temperature, min_p, top_p, max_tokens)
            if generate_type == "Pantella":
                new_characters = []
                for character in results["characters"]:
                    character["bio"] = character["final_description"]
                    del character["final_description"]
                    character["behavior_blacklist"] = []
                    character["behavior_whitelist"] = []
                    new_characters.append(character)
                results["characters"] = new_characters
            return results, schema_desc
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
        def download_results(results):
            with open('results.json', 'w') as f:
                json.dump(results, f, indent=2)
        if args.allow_deleting_entries:
            delete_entry_btn.click(gr_delete_entry, [entry_id], outputs=delete_results)
        search_btn.click(gr_search, [query, n_results, only_use_character_data_search], outputs=search_results)
        if args.allow_searching_generated_characters:
            search_generated_btn.click(search_generated, [generated_query, n_results_generated, only_use_character_data_search_generated], outputs=search_generated_results)
        generate_btn.click(gr_generate_character, [query_generate, n_results_generate, character_count, max_tokens, generate_type, temperature, min_p, top_p], outputs=[generate_results, schema_description])
        generate_lorebook_btn.click(gr_generate_lorebook_entry, [query_generate_lorebook, n_results_generate_lorebook, entry_count, max_tokens_lorebook, temperature_lorebook, min_p_lorebook, top_p_lorebook], outputs=generate_lorebook_results)
        download_btn.click(download_results, [generate_results])
            
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