import re
import argparse
import requests
from flask import request, jsonify

class DataProcessor:
    def filter_character_info(self, text: str) -> str:
        bullet_points = [
            "Full Name", "Nickname", "Age", "Gender", "Ethnicity", "Nationality",
            "Height", "Weight", "Hair Color", "Eye Color", "Scars or Tattoos",
            "Clothing Style", "Hobbies", "Favorite Food", "Language Proficiency",
            "Family", "Friends", "Education", "Occupational History", "Current Feelings and State",
            "Motivations", "Fears", "Secrets", "Past Year Overview",
            "Early life experiences", "Basic education", "Playmates", "Simple hobbies",
            "Educational aspirations", "Developing social skills", "Complex emotional states",
            "Higher education", "Early career experiences", "Romantic relationships",
            "Evolving personal beliefs", "Career progression", "Family dynamics",
            "Maturing worldviews", "Life achievements", "Major life changes",
            "Retirement or legacy thoughts"
        ]

        filtered_info = ""

        for point in bullet_points:
            match = re.search(f"{re.escape(point)}: ([^\n]*)", text)
            if match:
                filtered_info += f"{point}: {match.group(1)}\n"

        return filtered_info

class CharacterSeedGenerator:
    def __init__(self, processor, llm_service_url):
        self.processor = processor
        self.llm_service_url = llm_service_url

    def generate_text(self, prompt: str) -> str:
        data = {
            'prompt': prompt,
            'lora_params_path': None
        }
        response = requests.post(f"{self.llm_service_url}/generate", json=data)
        response_data = response.json()
        generated_text = response_data.get("generated_text", "")
        return generated_text

    def generate_character_seed(self, seed_prompt: str, temperature: float, max_tokens: int) -> str:
        core_character = self.generate_text(seed_prompt)
        print(f"CORE CHARACTER RAW: {core_character}")
        filtered_character = self.processor.filter_character_info(core_character)
        return filtered_character