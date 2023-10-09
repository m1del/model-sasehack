import os
import asyncio
import httpx
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Setup
cred = firebase_admin.credentials.Certificate('./firebase.json')
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Load variables from .env for Hugging Face API
load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/distilroberta-base"
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5

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
    return None

async def update_expirations_in_firestore():
    # Fetch items without 'daysTillExpire' attribute
    items_ref = db.collection(u'food').where(u'daysTillExpire', u'==', None).stream()

    for item in items_ref:
        data = item.to_dict()
        # Predict the expiration
        predicted_days = await predict_expiration(data['name'], data.get('location', 'fridge'), data.get('temperature', 'room temperature'))
        
        # Update Firestore
        if predicted_days is not None:  # only update if we got a valid prediction
            doc_ref = db.collection('food').document(item.id)
            doc_ref.update({"daysTillExpire": predicted_days})  


async def main():
    await update_expirations_in_firestore() 
    # food = input("Enter food: ")
    # location = input("Enter location: ")
    # temperature = input("Enter temperature: ")
    
    # predicted_days = await predict_expiration(food, location, temperature)
    # print(f"The {food} in the {location} will spoil after approximately {predicted_days} days.")

if __name__ == '__main__':
    asyncio.run(main())
