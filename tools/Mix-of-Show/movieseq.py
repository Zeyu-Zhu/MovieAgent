import time
import os
import json
import imageio
import requests
import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from utils import encode_image
from transformers import AutoModel, AutoTokenizer
import openai
from openai import OpenAI
import torch 
from PIL import Image
import vertexai

import IPython.display
from IPython.core.interactiveshell import InteractiveShell

# InteractiveShell.ast_node_interactivity = "all"

from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)
   



class MovieSeq_Char:
    def __init__(self,
                    model="gpt-4o", api_key=None, image_detail="auto",
                 system_text=None):
        self.api_key = api_key
        self.model = model
        self.image_detail = image_detail
        if system_text is None:
            self.system_text = """
        You will be provided with the following inputs:
        1. A photos of characters along with the names.

        Your task is to analyze and associate these inputs, and respond to the user's needs accordingly.
        """
            
        self.headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.api_key}"
        }
        self.url = "https://api.openai.com/v1/chat/completions"
        self.client = OpenAI()

    def get_response(self, name, char_url, 
                                query,
                                resize=None, temperature=0, detail="auto"):
        messages = [{
            "role": "system", 
            "content": [{"type": "text", "text": self.system_text,},]
            }]
        
        name = name.split("-")[-1].replace("_"," ")
        char_image = encode_image(char_url)
        messages.append({
            "role": "user",
            "content": [
                f"This is the photo of {name}.",
                {'image': char_image},
            ],
        })

        
        messages.append({
            "role": "user", 
            "content": [{"type": "text", "text": query,},]
        })
        
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": temperature,
        }
        
        response = self.client.chat.completions.create(**params)
        json_string = response.json()
        json_object = json.loads(json_string)
        content = json_object['choices'][0]['message']['content']
        return content
    
