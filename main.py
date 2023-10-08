import numpy as np
import pandas as pd

# Load the data
file_path = 'FoodKeeper-Data.csv'
data = pd.read_csv(file_path, encoding='latin1')

# Display basic information and first few rows of the data
data_info = data.info()
data_head = data.head()

data_info, data_head
