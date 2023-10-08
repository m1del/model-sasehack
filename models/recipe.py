import openai
import os
import readline
from dotenv import load_dotenv
load_dotenv()

# OpenAI config
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

def get_recipe(ingredients, inventory):
    prompt_text = (f"I have most cookware available at my disposal such as a stove, oven, microwave, air fryer, toaster, kettle, along with utensils like kitchen knives, spatulas, etc. "
                   f"I have these ingredients: {ingredients}. "
                   f"Give me a recipe that includes most, if not all of these ingredients. "
                   f"You may include other ingredients from my kitchen, such as {inventory} "
                   "and items that are common to most households, like salt, pepper, oregano, and garlic powder. "
                   "What can I make as well as the instructions?")

    response = openai.Completion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt_text}]
    )
    return response.choices[0].message["content"]

def main():
    ingredients = input("Enter your ingredients (comma-separated): ")
    inventory = input("Enter other items from your inventory (comma-separated): ")
    recipe = get_recipe(ingredients, inventory)
    print("\nRecipe and Instructions:\n", recipe)

if __name__ == "__main__":
    main()
