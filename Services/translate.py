import ollama
import json
import ast
import os
import re
from aksharamukha import transliterate

# def prepare_for_tts(text_list, target_code):
#     script_map = {
#         "gu": "Gujarati",
#         "ta": "Tamil",
#         "te": "Telugu",
#         "kn": "Kannada",
#         "ml": "Malayalam"
#     }
    
#     target_code = target_code.lower()
#     transliterated_list = []
    
#     for text in text_list:
#         for script in ['Urdu', 'Arabic', 'Bengali']:
#             text = transliterate.process(script, 'Devanagari', text)
        
#         if target_code in script_map:
#             source_script = script_map[target_code]
#             text = transliterate.process(source_script, 'Devanagari', text)
        
#         text = re.sub(r'[^\u0900-\u097F\u0020-\u007F\u2000-\u206F]', '', text)
#         transliterated_list.append(text)
        
#     return transliterated_list

def translate_list(filename, output_dir: str, json_data: str, TEXT_LIST_DATA, TARGET_CODE: str, GENDER: str, model_name="gemma4:e2b"):
    
    with open(json_data, "r", encoding="utf-8") as f:
        data = json.load(f)
  
    SOURCE_CODE = data[0]["audio-language"]
    TEXT = data[0]["transcription-list"]
    
    lang_mapping = {
        "hi": "Hindi", "en": "English", "gu": "Gujarati", "mr": "Marathi",
        "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam"
    }

    SOURCE_LANG = lang_mapping.get(SOURCE_CODE, SOURCE_CODE)
    TARGET_LANG = lang_mapping.get(TARGET_CODE, TARGET_CODE)

    # TIER 1 FIX: Aggressively enforced syllable matching limits to prevent physics-defying TTS stretches.
    system_prompt = f"""You are a Master {SOURCE_LANG} ({SOURCE_CODE}) to {TARGET_LANG} ({TARGET_CODE}) Dubbing Director and Translator. Your task is to translate a YouTube Tech/Product Review video transcript for Text-to-Speech (TTS) dubbing. 

SPECIAL CONTEXT FOR VIDEO DUBBING:
The input is an ASR (Automatic Speech Recognition) transcript formatted as a Python list of dictionaries. 
CRITICAL NOTE: ASR chunks audio purely by timestamps and pauses. A single sentence, joke, or thought often spans multiple chunks. Furthermore, ASR lacks punctuation. You must understand the speaker's overarching emotion, suspense, and colloquial tech-reviewer tone before translating.

🧠 MENTAL WORKFLOW (Internal reasoning before generating output):
1. READ & PUNCTUATE: Mentally read the entire input array. Add punctuation to understand the context, sarcasm, and true meaning of the source text. Do not translate blindly word-for-word.
2. ADAPT THE TONE: Tech reviewers use suspense ("this is supposed to be the most..."), colloquialisms, and quick pacing. Recreate this exact emotion and energy using natural {TARGET_LANG} phrasing and idioms.
3. ADAPT THE GRAMMAR (NO CASCADING DESYNC): Understand that sentence structures differ (e.g., English SVO vs. Hindi/Gujarati SOV). You must adapt the sentence structure to sound perfectly natural in {TARGET_LANG}, BUT you must split the resulting sentence back into the exact same timeframes. 

🔴 ZERO-TOLERANCE CONSTRAINTS 🔴
1. EXACT LIST LENGTH: You MUST return a valid Python list of strings containing the EXACT SAME NUMBER OF ELEMENTS as the input list.
2. STRICT TIMELINE SYNC (ANTI-BLEEDING): You are allowed to shift words *slightly* between adjacent chunks to make the grammar sound natural, but you MUST NEVER cascade translations down the array. Index [6] must translate the core concept happening at Input [6]'s timestamp. Never push Index [6]'s translation into Index [7].
3. STRICT DURATION MATCHING: The spoken {TARGET_LANG} text MUST naturally fit the exact "duration" provided for that specific chunk. If a target translation is too long for a 1.5s duration, concisely summarize it. If it is too short for a 4.0s duration, expand it naturally.
4. NATURAL VOCABULARY: Do not use overly formal/literary words. Keep common tech nouns in English but use phonetic spelling in the {TARGET_LANG} script (e.g., "bloatware/specs/smartphone" -> "બ્લોટવેર/સ્પેક્સ/સ્માર્ટફોન" in Gujarati). Reflect that the speaker is {GENDER}.

🌟 COMPREHENSIVE UNDERSTANDING EXAMPLE 🌟
The following is an example demonstrating how to extract meaning, adapt grammar, and maintain strict timestamp synchronization. DO NOT copy this text; use it to understand the required logic.

JUST EXAMPLE Input (English to Gujarati Tech Review):
[
    {{"en": "this right here is the brand new nothing phone 3a lite and this is supposed to be the most", "duration": 3.92}},
    {{"en": "affordable nothing phone till date now i know what you're thinking isn't that the cmf well i", "duration": 5.12}},
    {{"en": "know the cmf phone 2 pro is a great budget smartphone but at the end of the day it's a", "duration": 4.0}},
    {{"en": "cmf phone this on the other hand is supposed to be more nothing you subscribe for more", "duration": 4.36}},
    {{"en": "made-up words and i know there's a lot of questions is this actually a good", "duration": 2.6}},
    {{"en": "Great budget, nothing for you to buy in India.", "duration": 1.68}},
    {{"en": "What about the specs?", "duration": 0.74}},
    {{"en": "How is it different to the CMF?", "duration": 1.36}},
    {{"en": "Does it come with bloatware?", "duration": 1.64}},
    {{"en": "And if yes, how much?", "duration": 0.88}},
    {{"en": "Time to answer all of your questions.", "duration": 1.4}}
]

Correct Output Logic (Notice how the tone is maintained, grammar is native, and indices NEVER cascade out of sync):
[
    "આ જે તમે જોઈ રહ્યા છો, તે છે બ્રાન્ડ ન્યૂ Nothing Phone 3a Lite. અને એવું માનવામાં આવે છે કે આ અત્યાર સુધીનો...", 
    "...સૌથી સસ્તો Nothing ફોન છે. હવે, મને ખબર છે કે તમે શું વિચારી રહ્યા છો, શું એ CMF ફોન નથી? તો જુઓ, હું...", 
    "...જાણું છું કે CMF Phone 2 Pro એક કમાલનો બજેટ સ્માર્ટફોન છે. પણ દિવસના અંતે, એ છે તો એક...",
    "...CMF ફોન જ ને. બીજી બાજુ, આ ફોન તમને 'Nothing' નો અસલી અનુભવ આપશે. આવા જ નવા નવા...",
    "...શબ્દો સાંભળવા માટે સબ્સ્ક્રાઇબ કરો! મને ખબર છે કે તમારા મનમાં ઘણા સવાલો હશે. શું આ ખરેખર એક સારો...",
    "...બજેટ Nothing ફોન છે, જે તમારે ભારતમાં લેવો જોઈએ?",
    "આના સ્પેક્સ કેવા છે?",
    "આ CMF કરતા કઈ રીતે અલગ છે?",
    "શું આમાં બ્લોટવેર જોવા મળશે?",
    "અને જો હા, તો કેટલું?",
    "તમારા બધા સવાલોના જવાબ આપવાનો સમય આવી ગયો છે."
]

The above example was to Translate English to Gujarati. You must follow the exact same logic for any language pair. Current language pair is {SOURCE_LANG} to {TARGET_LANG} translation.

GENERAL RULES:
- Phonetic Transliteration: Convert brand names/acronyms into spoken phonetic spellings in {TARGET_LANG} (e.g., "S24" -> "એસ ટ્વેન્ટી ફોર" (if translation from English to Gujarati)).
- Number Conversion: Numbers MUST be strictly converted to English pronunciation written in words in the {TARGET_LANG} script (e.g., "100" -> "વન હન્ડ્રેડ" (if translation from English to Gujarati)). Do NOT output digits.
- Output Format: Strictly output ONLY the raw Python list of strings. No markdown formatting, no dictionary keys, no conversational filler.

EXPECTED OUTPUT FORMAT:
["Translation 0", "Translation 1", "Translation 2", ...]

Please process the following data:
{TEXT_LIST_DATA}
"""
    
    print(system_prompt)

    response = []
    
    try:
        raw_response = ollama.chat(
            model=model_name, messages=[{"role":"user", "content":system_prompt}]
        )["message"]["content"]
        
        response = ast.literal_eval(raw_response)
        print(response)
        
        if len(response) != len(TEXT):
            print(f"Length mismatch: {len(response)} != {len(TEXT)}. Retrying...")
            return translate_list(filename, output_dir, json_data, TEXT_LIST_DATA, TARGET_CODE, GENDER, model_name)
        
        # response = prepare_for_tts(response, TARGET_CODE)
        print("\n", response)
            
    except Exception as e:
        print(f"Error during translation: {e}. Retrying...")
        return translate_list(filename, output_dir, json_data, TEXT_LIST_DATA, TARGET_CODE, GENDER, model_name)

    with open(json_data, "r", encoding="utf-8") as file:
        data = json.load(file)

    for i in range(len(response)):
        data[0]["transcription"][i][f"{TARGET_CODE}"] = response[i]

    with open(json_data, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
        
    return 