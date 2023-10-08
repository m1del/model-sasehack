from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForMaskedLM.from_pretrained("distilroberta-base")

def predict_expiration(model, tokenizer, food_item, location, temperature):
    # Constructing the input sentence
    input_sentence = (
        f"I am storing {food_item} in my {location} "
        f"at {temperature}°F, how long until it spoils or goes bad?"
    )
    
    # Tokenizing the input sentence
    inputs = tokenizer(input_sentence, return_tensors='pt', padding=True, truncation=True, max_length=64)
    
    # Moving inputs to the same device as the model
    inputs = {name: tensor.to(next(model.parameters()).device) for name, tensor in inputs.items()}
    
    # Get the model output
    with torch.no_grad():
        output = model(**inputs)
    
    # Convert model output to days (as a single float value)
    predicted_days = output.item()
    
    return predicted_days

def main():
    food = input("Enter food: ")
    location = input("Enter location: ")
    temperature = input("Enter temperature: ")

    predicted_days = predict_expiration(model, tokenizer, food, location, temperature)
    print(f"The {food} in the {location} will spoil after approximately {predicted_days:.2f} days.")

if __name__ == '__main__':
    main()