import os
import asyncio
import httpx
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/distilroberta-base"
API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {"Authorization": f"Bearer {API_TOKEN}"}

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5

async def query(payload):
    async with httpx.AsyncClient() as client:
        for attempt in range(MAX_RETRIES):
            response = await client.post(API_URL, headers=headers, json=payload)
            response_json = response.json()
            
            # print(f"API Response: {response_json}")
            
            if response.status_code == 200 and 'error' not in response_json:
                return response_json
            elif 'estimated_time' in response_json:
                await asyncio.sleep(response_json['estimated_time'])
            else:
                await asyncio.sleep(RETRY_WAIT_SECONDS)
        raise ValueError("Max retries reached. API request failed with status code " + str(response.status_code))

async def predict_expiration(food_item, location, temperature):
    input_sentence = (
        f"I am storing {food_item} in my {location} "
        f"at {temperature} degrees fahrenheit, it will spoil in <mask> days."
    )
    predictions = await query({"inputs": input_sentence})
    
    # Find the first numerical prediction
    for prediction in predictions:
        token_str = prediction['token_str'].strip()
        if token_str.isdigit():
            return token_str
    
    # null checksss
    return "an unknown number of"


async def main():
    food = input("Enter food: ")
    location = input("Enter location: ")
    temperature = input("Enter temperature: ")
    
    predicted_days = await predict_expiration(food, location, temperature)
    print(f"The {food} in the {location} will spoil after approximately {predicted_days} days.")

if __name__ == '__main__':
    asyncio.run(main())
