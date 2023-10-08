import os
import asyncio
import httpx
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore

# Firebase Setup
cred = firebase_admin.credentials.Certificate('./firebase.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Load variables from .env for Hugging Face API
load_dotenv()
API_URL = "https://api-inference.huggingface.co/models/distilroberta-base"
API_TOKEN = os.getenv("HF_API_TOKEN")
headers = {"Authorization": f"Bearer {API_TOKEN}"}
MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 5

db = firestore.client()

def get_data_from_firestore(collection_name):
    docs = db.collection(collection_name).stream()
    data = []
    for doc in docs:
        entry = doc.to_dict()
        entry['doc_id'] = doc.id
        data.append(entry)
    return data

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
    return "unknown"


def update_firestore_with_predictions(collection_name, data):
    for entry in data:
        doc_ref = db.collection(collection_name).document(entry['doc_id'])
        doc_ref.update({"daysTillExpire": entry['predicted_expiration']})
        
def process_batch(batch):
    # We will make async calls in this function
    async def _process_batch():
        for entry in batch:
            food = entry['name']
            location = entry['location']
            temperature = str(entry.get('temperature', ''))  # Convert temperature to string

            predicted_days = await predict_expiration(food, location, temperature)
            entry['predicted_expiration'] = predicted_days

    asyncio.run(_process_batch())
    
def print_pretty_data(data):
    # Print headers
    print('-'*90)  # Line separator
    print(f"{'Document ID':<25} | {'Food':<15} | {'Location':<10} | {'Days Till Expire'}")
    print('-'*90)  # Line separator
    
    # Print each data entry
    for entry in data:
        doc_id = entry['doc_id']
        name = entry['name']
        location = entry['location']
        days_till_expire = entry.get('daysTillExpire', 'Unknown')  # Use 'Unknown' if not available
        
        print(f"{doc_id:<25} | {name:<15} | {location:<10} | {days_till_expire}")

    print('-'*90)  # Line separator
        
def main():
    # Fetch data from Firestore
    data = get_data_from_firestore('food')
    
    # print_pretty_data(data)
	# Processes batch
    process_batch(data)

    # Update Firestore with the predicted expiration dates
    update_firestore_with_predictions('food', data)


if __name__ == '__main__':
    main()