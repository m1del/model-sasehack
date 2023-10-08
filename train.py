import pandas as pd
import re
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim

class BertRegressor(nn.Module):
    def __init__(self):
        super(BertRegressor, self).__init__()
        self.bert = RobertaModel.from_pretrained('distilroberta-base')
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)  # Regression head
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_output)
    

def convert_to_days(time_str):
    try:
        times = [float(t) for t in re.split('-| ', time_str) if t.replace('.', '').isdigit()]
        
        if "-" in time_str and len(times) == 2:
            avg_time = sum(times) / 2
        elif len(times) == 1:
            avg_time = times[0]
        else:
            return None
        
        # Convert to days
        if "week" in time_str:
            avg_time *= 7  # convert weeks to days
        elif "month" in time_str:
            avg_time *= 30  # convert months to days
        elif "year" in time_str:
            avg_time *= 365.25
        
        return avg_time
    except Exception as e:
        print(f"Exception during conversion: {str(e)}") 
        return None

def load_data(file_path):
    return pd.read_csv(file_path)

def process_data(data):
    # Extract relevant columns related to expiration and location
    relevant_columns_with_temp = [
        ("Pantry_Text", "pantry", 60),
        ("Refrigerate_Text", "refrigerator", 40),
        ("DateOfPurchase_Freeze_Text", "freezer", 0),
    ]
    
    # Input and Output Sentences
    input_sentences_with_temp = []
    output_days_with_temp = []
    
    for _, row in data.iterrows():
        for col, location, temp in relevant_columns_with_temp:
            # Check if expiration data is available
            if pd.notna(row[col]):
                # Construct the input sentence with temperature
                input_sentence = (
                    f"I am storing {row['Name']} in my {location} "
                    f"at {temp}°F, how long until it spoils or goes bad?"
                )
                output_day = convert_to_days(row[col])
                # Append to respective lists if output_day is not None
                if output_day is not None:
                    input_sentences_with_temp.append(input_sentence)
                    output_days_with_temp.append(output_day)
                else:
                    # Print the data point that is returning None in convert_to_days
                    print(f"Conversion Issue: {row[col]}")
                    
    #Debugging
    # for i in range(5):
    #     print("Input Sentence:")
    #     print(input_sentences_with_temp[i])
    #     print("Output Sentece:")
    #     print(output_days_with_temp [i])

    
    # Tokenizing
    tokenizer = RobertaTokenizer.from_pretrained('distilroberta-base')

    # Restrict the max length to 64 for computational efficiency; adjust as needed
    max_length = 64
    input_encodings = tokenizer(input_sentences_with_temp, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    # Debugging
    print(f"Num input sentences: {len(input_sentences_with_temp)}")
    print(f"Num output days: {len(output_days_with_temp)}")



    # Extract input ids and attention masks as PyTorch tensors
    input_ids = input_encodings['input_ids']
    attention_masks = input_encodings['attention_mask']

    # Convert output (days) to PyTorch tensor
    output_days_tensor = torch.tensor(output_days_with_temp).view(-1, 1)
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Attention Masks shape: {attention_masks.shape}")
    print(f"Output Days Tensor shape: {output_days_tensor.shape}")

    
    # Load into a dataset
    dataset = TensorDataset(input_ids, attention_masks, output_days_tensor)
    return dataset, tokenizer

def initialize_model():
    model = BertRegressor()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # Adjust the learning rate as needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, optimizer, device

def train_model(model, optimizer, train_dataloader, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in train_dataloader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f"Epoch: {epoch+1}, Loss: {total_loss/len(train_dataloader)}")
        

def validate_model(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask)
            loss = nn.MSELoss()(outputs, labels.float())
            total_loss += loss.item()
    print(f"Validation Loss: {total_loss/len(val_dataloader)}")
    
    
def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main():
    set_seed()
    raw_data = load_data('./data/cleaned_data.csv')
    data, tokenizer = process_data(raw_data)
    
    # Split data, train and val sets
    train_size = int(0.9 * len(data))
    val_size = len(data) - train_size   
    # mf sanity check pls help me
    if train_size == 0 or val_size == 0:
        raise ValueError("Insufficient data for splitting into training and validation sets.")
    
    train_dataset, val_dataset = random_split(data, [train_size, val_size])
    
    
    # Initialize model, optimizer, and device
    model, optimizer, device = initialize_model()

    num_epochs = 10

    # Train and Validate
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    train_model(model, optimizer, train_dataloader, device, num_epochs)
    validate_model(model, val_dataloader, device)
    
    # export ?? :sob:
    torch.save(model.state_dict(), './expiration.pth')
    tokenizer.save_pretrained('./expiration_tokenizer')
    
if __name__ == "__main__":
    main()