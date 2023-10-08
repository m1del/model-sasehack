# Food Expiry Date Prediction

This project utilizes a deep learning model to predict the expiry date of a food item based on its storage location. Given a food item and a location (pantry, refrigerator, or freezer), the model estimates the number of days until the item spoils or goes bad.

## Overview

Understanding the shelf-life of food items is crucial for both minimizing waste and ensuring that the consumed items are fresh and safe. This project aims to create a model that can predict the expiration date of a food item in a specific storage condition, providing users with a tool to better manage their food storage and consumption.

The model is built upon the BERT (Bidirectional Encoder Representations from Transformers) architecture, which has been successful in various Natural Language Processing (NLP) tasks.

## Data

To be honest, the [data](https://www.fsis.usda.gov/shared/data/EN/FoodKeeper-Data.xls) is quite bad. I made use of the [USDA's Foodkeeper](https://www.foodsafety.gov/keep-food-safe/foodkeeper-app) app, well whatever they provided in the free dataset.
The dataset includes various food items along with their respective storage instructions and expiry details for different storage locations, namely the pantry, refrigerator, and freezer, and tips on how/where to store it. There isn't much data, and it's missing a lot of values. Had to make do though </3.

## Model

The BertRegressor model, a custom PyTorch model, utilizes a pre-trained BERT variant for predicting a continuous value (number of days until the food spoils). Given an input sentence that describes the food item and its storage location, the model outputs a single scalar value representing the predicted number of days until expiration.

## Setup and Installation
### Requirements
- Python 3.7+
- Pytorch 1.8+
- Transformers 4.0
- A CUDA 12.x enabled GPU if using our preconfigured

### Installation
1. Clone the repository
  ```
  git clone
  cd model-sasehack
  ```
2. Install the requirements
   ```
   conda install -r requirements.txt
   ```

### Usage
- Run the training script, it will produce a model with the training weights as `expiration.pth`, and a tokenizer dir `expiration_tokenizer`/
  ```
  python train.py
  ```
## Example
### Input
- Food: Chicken
- Location: Refrigerator
- Temperature: 35
  (input number only, I'm assuming imperial #GoAmerica)
### Query to the Model
"I am storing chicken in my refrigerator at 35F, how long until it spoils or goes bad?"
### Output (as of when I'm writing this lol)
11.97 days

## Future Work
- The data is quite lacking, could be improved by a lot with quality diverse training data.
