import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

def predict_expiration(model, tokenizer, food_item, location, temperature):
    # Constructing the input sentence
    input_sentence = (
        f"I am storing {food_item} in my {location} "
        f"at {temperature} degrees fahrenheit, it will spoil in <mask> days."
    )
    # Tokenizing
    inputs = tokenizer(input_sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {name: tensor.to(next(model.parameters()).device) for name, tensor in inputs.items()}
    
    # Get the model output
    with torch.no_grad():
        output = model(**inputs)
    
    # Extracting prediction and convert
    predicted_token_id = torch.argmax(output.logits[0, inputs['input_ids'][0] == tokenizer.mask_token_id])
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]
    predicted_days = ''.join(filter(str.isdigit, predicted_token))
    
    # Null checkssss
    if not predicted_days:
        predicted_days = "an unknown number of"
    
    return predicted_days

def main():
    food = input("Enter food: ")
    location = input("Enter location: ")
    temperature = input("Enter temperature: ")

    predicted_days = predict_expiration(model, tokenizer, food, location, temperature)
    print(f"The {food} in the {location} will spoil after approximately {predicted_days} days.")

if __name__ == '__main__':
    main()
