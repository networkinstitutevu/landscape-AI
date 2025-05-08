import json
import os
import requests
from dotenv import load_dotenv

load_dotenv('.env')

NEBULA_URL='https://nebula.cs.vu.nl/api/chat/completions'
NEBULA_TOKEN=str(os.getenv('NEBULA_TOKEN'))

def chat_with_model(model, system_prompt, user_prompt):
    url = NEBULA_URL
    headers = {
        'Authorization': f'Bearer {NEBULA_TOKEN}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                ]
            },
        ]
    }
 
    response = requests.post(url, headers=headers, json=data)
    return response.json()


if __name__=='__main__':
    print(chat_with_model('llama3.1:8b', 'You are a helpful assistant', 'How are you today?'))
