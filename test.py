import torch
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import logging


logging.basicConfig(level=logging.ERROR)

class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = RobertaModel.from_pretrained('distilroberta-base')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # Regression head
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output)   

# Load the model and tokenizer
model = BertRegressor()
model.load_state_dict(torch.load('./models/expiration.pth', map_location=torch.device('cpu')))
model.eval()
tokenizer = RobertaTokenizer.from_pretrained('./models/expiration_tokenizer/')

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
